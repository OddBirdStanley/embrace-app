"""
_train(model, sig, label) - 

Notes:
1. _train currently takes in one batch at a time.
2. Training is always in place. Whether this model is saved as a new file is not dealt with here.
"""
import numpy as np
from typing import Tuple, Dict
import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import math
import copy
from sklearn.metrics import f1_score
import torch.nn as nn
import pandas as pd
from pathlib import Path
import re
from dataclasses import dataclass

METADATA_PATH = Path("artifacts/metadata_v10.json")

@dataclass
class PreprocessingConfig:
    """
    Stores preprocessing and windowing settings for the new Mindrove CSV files.
    """

    # 200 samples at 500 Hz = 0.40 seconds of EMG history per prediction.
    window_size: int = 200

    # 25 samples at 500 Hz = one training window every 0.05 seconds.
    step_size: int = 25

    # Mindrove armband sampling rate.
    sample_rate_hz: int = 500

    # Ignore labeled segments that are too short to be useful.
    min_segment_seconds: float = 0.75

    # Optional safety filter: remove non-Rest windows whose centered RMS energy is too close to Rest.
    gesture_energy_multiplier: float = 1.10
    use_gesture_energy_filter: bool = False

SPEC_LABEL_MAP = {
    "Extend": 0,
    "Fist": 1,
    "Flex": 2,
    "Pro": 3,
    "Radial": 4,
    "Rest": 5,
    "Sup": 6,
    "Ulnar": 7,
}
SPEC_INDEX_TO_LABEL = {v: k for k, v in SPEC_LABEL_MAP.items()}

cfg = PreprocessingConfig()
min_segment_samples = int(cfg.min_segment_seconds * cfg.sample_rate_hz)

def create_df(signals):
    """
    Create a pandas DataFrame from signals np array
    """
    df = pd.DataFrame(signals)
    df = df.dropna(how="all").reset_index(drop=True)
    
    num_channels = df.shape[1] - 1
    channel_cols = [f"Channel{i + 1}" for i in range(num_channels)]
    label_col = "label_value"

    df.columns = channel_cols + [label_col]

    for col in channel_cols + [label_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_channel_rows = df[channel_cols].isna().any(axis=1).sum()
    if bad_channel_rows > 0:
        raise ValueError(
            f"Input signals have {bad_channel_rows} rows with missing/non-numeric channel values. "
            "The label column may be NaN, but channel columns should not be NaN."
        )
    
    return df

def label_value_to_name(value) -> object:
    """
    Convert numeric labels such as 0.0, 1.0, ..., 7.0 into class names.
    """
    if pd.isna(value):
        return np.nan

    # Convert to float first to handle cases where the label might be read as a string or a float.
    value_float = float(value)
    label_id = int(value_float) if value_float.is_integer() else value_float

    # Map the label ID to a name using the SPEC_INDEX_TO_LABEL dictionary
    return SPEC_INDEX_TO_LABEL.get(label_id, f"Label_{label_id}")

def infer_channel_cols(df: pd.DataFrame) -> list[str]:
    """
    Dynamically find channel columns named Channel1, Channel2, ...
    """

    # Use a regex to find columns that match the pattern "Channel" followed by a number.
    channel_cols = [col for col in df.columns if re.fullmatch(r"Channel\d+", str(col))]

    # Sort the channel columns in numerical order based on the number in their name.
    def sort_key(name: str):
        m = re.search(r"(\d+)$", name)
        return int(m.group(1)) if m else float("inf")

    return sorted(channel_cols, key=sort_key)

#Re-segment and Window Functions
def compute_emg_energy(segment: np.ndarray) -> float:
    """
    Compute centered RMS energy for an EMG window.
    """
    if len(segment) == 0:
        return 0.0
    
    # Compute the centered RMS energy across all channels and return the average.
    segment = segment.astype(np.float32)
    centered = segment - np.mean(segment, axis=0, keepdims=True)
    rms_per_channel = np.sqrt(np.mean(np.square(centered), axis=0))
    return float(np.mean(rms_per_channel))

def add_labeled_segment_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a segment ID to each contiguous non-NaN labeled region.
    """
    df = df.copy()

    segment_ids = []
    current_segment_id = -1
    previous_label = None

    # Iterate through the label column and assign segment IDs to contiguous labeled regions.
    for label in df["label"].tolist():
        # If the label is NaN, assign NaN to segment_id and reset previous_label.
        if pd.isna(label):
            segment_ids.append(np.nan)
            previous_label = None
            continue
        
        # If the label is different from the previous label, start a new segment.
        if previous_label is None or label != previous_label:
            current_segment_id += 1

        # Assign the current segment ID to this row.
        segment_ids.append(current_segment_id)
        previous_label = label

    df["segment_id"] = segment_ids
    return df

def extract_labeled_segments(
    df: pd.DataFrame,
    channel_cols: list[str],
    cfg: PreprocessingConfig
) -> list[pd.DataFrame]:
    """
    Extract contiguous labeled gesture/rest segments from one file.
    Pause rows with NaN labels are ignored.
    """
    df = add_labeled_segment_ids(df)
    segments = []

    # Only keep rows with valid segment IDs (non-NaN) for segment extraction.
    labeled_df = df[df["segment_id"].notna()].copy()
    if len(labeled_df) == 0:
        return segments

    # Group by segment_id and extract segments that are long enough based on the config settings.
    for segment_id, segment_df in labeled_df.groupby("segment_id", sort=True):
        # Reset the index of the segment DataFrame for easier processing later.
        segment_df = segment_df.reset_index(drop=True)

        # Only keep segments that are long enough to create at least one training window based on the config settings.
        if len(segment_df) < min_segment_samples:
            continue
        
        # Add segment_id and segment_label columns to the segment DataFrame for later reference.
        segment_df["segment_id"] = int(segment_id)
        segment_df["segment_label"] = str(segment_df["label"].iloc[0])

        # Append the valid segment DataFrame to the list of segments.
        segments.append(segment_df)

    return segments

def window_start_indices(length: int, window_size: int, step_size: int) -> list[int]:
    """
    Return sliding-window start indices for one segment.
    """

    # If the segment is too short to fit even one window, return an empty list.
    if length < window_size:
        return []

    # Calculate the start indices for sliding windows based on the window size and step size.
    return list(range(0, length - window_size + 1, step_size))

def make_windows_from_segment(
    segment_df: pd.DataFrame,
    channel_cols: list[str],
    cfg: PreprocessingConfig,
    rest_energy_threshold: float | None = None,
    gesture_energy_multiplier: float = 1.10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from one labeled segment.

    Each output window has shape:
    (window_size, num_channels)
    """

    # Extract the EMG data for the segment and convert to a NumPy array.
    data = segment_df[channel_cols].to_numpy(dtype=np.float32)

    # Get the label for this segment from the first row (all rows in the segment have the same label).
    label = str(segment_df["label"].iloc[0])

    T, C = data.shape
    starts = window_start_indices(T, cfg.window_size, cfg.step_size)

    # If there are no valid window start indices, return empty arrays.
    if len(starts) == 0:
        return (
            np.empty((0, cfg.window_size, C), dtype=np.float32),
            np.empty((0,), dtype=object)
        )

    kept_starts = []

    # Optional filter for mislabeled gesture windows that look too much like Rest.
    for s in starts:
        window = data[s:s + cfg.window_size]

        if label != "Rest" and rest_energy_threshold is not None:
            energy = compute_emg_energy(window)
            if energy < gesture_energy_multiplier * rest_energy_threshold:
                continue

        kept_starts.append(s)

    # If no windows were kept after filtering, return empty arrays.
    if len(kept_starts) == 0:
        return (
            np.empty((0, cfg.window_size, C), dtype=np.float32),
            np.empty((0,), dtype=object)
        )

    # Stack the kept windows into a single NumPy array and create a corresponding label array.
    X = np.stack(
        [data[s:s + cfg.window_size] for s in kept_starts],
        axis=0
    ).astype(np.float32)

    # Create a label array of the same length as the number of kept windows, filled with the segment's label.
    y = np.array([label] * len(kept_starts), dtype=object)

    return X, y

#train split functions
def split_by_group_class_aware(
    groups: np.ndarray,
    labels: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
    max_tries: int = 1000
) -> tuple[set[str], set[str], set[str]]:
    """
    Split windows by group while trying to preserve class coverage.
    """
    rng = np.random.default_rng(random_state)

    unique_groups = np.unique(groups)
    all_classes = set(np.unique(labels).tolist())

    if len(unique_groups) < 3:
        raise ValueError(
            f"Need at least 3 split groups for train/val/test, but found {len(unique_groups)}. "
            "Add more CSV recordings or temporarily use a train/val split only."
        )

    best_split = None
    best_score = -np.inf

    for _ in range(max_tries):
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)

        n_groups = len(shuffled_groups)

        n_train = max(1, int(np.floor(train_size * n_groups)))
        n_val = max(1, int(np.floor(val_size * n_groups)))

        # Ensure test receives at least one group.
        if n_train + n_val >= n_groups:
            n_train = max(1, n_groups - 2)
            n_val = 1

        train_groups = set(shuffled_groups[:n_train])
        val_groups = set(shuffled_groups[n_train:n_train + n_val])

        train_mask_tmp = np.array([g in train_groups for g in groups])
        val_mask_tmp = np.array([g in val_groups for g in groups])

        train_labels = set(labels[train_mask_tmp])
        val_labels = set(labels[val_mask_tmp])

        score = 0

        # Strongly prefer train having all classes.
        if train_labels == all_classes:
            score += 1000
        else:
            score += 50 * len(train_labels)

        # Also prefer validation/test to cover as many classes as possible.
        score += 10 * len(val_labels)

        # Prefer split proportions close to requested sizes.
        score -= abs(len(train_groups) / n_groups - train_size)
        score -= abs(len(val_groups) / n_groups - val_size)

        if score > best_score:
            best_score = score
            best_split = (train_groups, val_groups)

    if best_split is None:
        raise RuntimeError("Could not create a valid group split.")

    return best_split

#normalization functions
def remove_window_dc_offset(X: np.ndarray) -> np.ndarray:
    """
    Remove the per-window, per-channel baseline.

    X has shape:
    (num_windows, window_size, num_channels)
    """
    window_mean = np.mean(X, axis=1, keepdims=True)
    return (X - window_mean).astype(np.float32)

def add_delta_features(X: np.ndarray) -> np.ndarray:
    """
    Add first-difference features.
    """
    delta = np.diff(X, axis=1, prepend=X[:, :1, :])
    return np.concatenate([X, delta], axis=2).astype(np.float32)

def prepare_model_features(X: np.ndarray) -> np.ndarray:
    """
    Apply all feature engineering steps before standardization.
    """
    X_centered = remove_window_dc_offset(X)
    X_features = add_delta_features(X_centered)
    return X_features.astype(np.float32)

def fit_channel_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a channel-wise standardizer using only the training set.
    """
    flat = X.reshape(-1, X.shape[-1])

    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0) + 1e-8

    return mean.astype(np.float32), std.astype(np.float32)

def apply_channel_standardization(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Apply training-set standardization to train/val/test data.
    """
    return ((X - mean[None, None, :]) / std[None, None, :]).astype(np.float32)

#encode functions
def encode_labels(labels: np.ndarray, label_map: Dict[str, int]) -> np.ndarray:
    """
    Encode string labels into integer class IDs.
    """
    return np.array([label_map[label] for label in labels], dtype=np.int64)

#loader functions/class
class EMGWindowsDataset(Dataset):
    """
    A PyTorch Dataset for EMG windows.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.01,
        gain_jitter_std: float = 0.05
    ):
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        self.augment = augment
        self.noise_std = noise_std
        self.gain_jitter_std = gain_jitter_std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            # Add very small Gaussian noise
            x = x + torch.randn_like(x) * self.noise_std

            # Add very small per-channel amplitude jitter
            gains = 1.0 + torch.randn(x.shape[1]) * self.gain_jitter_std
            gains = gains.to(x.device)
            x = x * gains.unsqueeze(0)

        return x, y

def prepare_loaders(sig: np.ndarray):
    
    #these two variables are copied from the metadata
    label_map = {
        "Extend": 0,
        "Fist": 1,
        "Flex": 2,
        "Pro": 3,
        "Radial": 4,
        "Rest": 5,
        "Sup": 6,
        "Ulnar": 7
    }
    engineered_feature_names = [
        "Channel1",
        "Channel2",
        "Channel3",
        "Channel4",
        "Channel5",
        "Channel6",
        "Channel7",
        "Channel8",
        "Channel1_delta",
        "Channel2_delta",
        "Channel3_delta",
        "Channel4_delta",
        "Channel5_delta",
        "Channel6_delta",
        "Channel7_delta",
        "Channel8_delta"
    ]

    df = create_df(sig)
    file_channel_cols = infer_channel_cols(df)

    df["label"] = df["label_value"].apply(label_value_to_name)
    df["source_file"] = "Live Signals"
    df["source_index"] = np.arange(len(df), dtype=np.int64)
    df["time_seconds"] = df["source_index"] / cfg.sample_rate_hz
    df["subject_id"] = "The User"

    df = df[
        ["source_file", "subject_id", "source_index", "time_seconds"]
        + file_channel_cols
        + ["label_value", "label"]
    ].copy()

    ft_schemas = []

    ft_schemas.append({
        "file": "Live Signals",
        "subject_id": df["subject_id"].iloc[0],
        "num_rows": len(df),
        "n_channels": len(file_channel_cols),
        "channels": file_channel_cols,
        "labeled_rows": int(df["label"].notna().sum()),
        "pause_rows": int(df["label"].isna().sum()),
    })

    if len(df) == 0:
        raise FileNotFoundError(
            f"No valid data rows found in the input signals."
        )

    schemas_df = pd.DataFrame(ft_schemas)

    channel_cols = infer_channel_cols(df)

    if len(channel_cols) == 0:
        raise ValueError("No channel columns were detected.")

    channel_counts = schemas_df["n_channels"].unique()
    if len(channel_counts) != 1:
        raise ValueError(
            "Not all files have the same number of channels. "
            f"Detected channel counts: {sorted(channel_counts.tolist())}"
        )
    
    filtered_df = df.copy()
    all_segments = []

    for source_file, sub in filtered_df.groupby("source_file", sort=False):
        sub = sub.reset_index(drop=True)
        file_segments = extract_labeled_segments(sub, channel_cols, cfg)
        all_segments.extend(file_segments)

    segment_summary = pd.DataFrame([
        {
            "source_file": segment_df["source_file"].iloc[0],
            "segment_id": int(segment_df["segment_id"].iloc[0]),
            "label": segment_df["segment_label"].iloc[0],
            "samples": len(segment_df),
            "seconds": len(segment_df) / cfg.sample_rate_hz,
        }
        for segment_df in all_segments
    ])

    rest_energy_threshold = None

    if cfg.use_gesture_energy_filter:
        rest_window_energies = []
        
        # Compute the energy of all Rest windows across all segments to determine a threshold for filtering gesture windows.
        for segment_df in all_segments:
            label = str(segment_df["segment_label"].iloc[0])

            if label != "Rest":
                continue

            data = segment_df[channel_cols].to_numpy(dtype=np.float32)
            starts = window_start_indices(len(data), cfg.window_size, cfg.step_size)

            for s in starts:
                window = data[s:s + cfg.window_size]
                rest_window_energies.append(compute_emg_energy(window))

        # Compute the 95th percentile Rest energy threshold to use for filtering gesture windows.
        if len(rest_window_energies) > 0:
            rest_energy_threshold = float(np.percentile(rest_window_energies, 95))
        else:
            rest_energy_threshold = None

        print("\nRest windows used to estimate threshold:", len(rest_window_energies))
        print("95th percentile Rest energy threshold:", rest_energy_threshold)
    else:
        print("\nGesture energy filter disabled.")
        print("No gesture windows will be removed for looking too similar to Rest.")

    X_list = []
    y_list = []
    file_list = []
    subject_list = []
    segment_ids = []

    dropped_segment_count = 0
    kept_window_count = 0

    for segment_df in all_segments:
        expected_label = str(segment_df["segment_label"].iloc[0])
        segment_id = int(segment_df["segment_id"].iloc[0])

        # Create sliding windows from this segment
        X_segment, y_segment = make_windows_from_segment(
            segment_df,
            channel_cols,
            cfg,
            rest_energy_threshold=rest_energy_threshold,
            gesture_energy_multiplier=cfg.gesture_energy_multiplier
        )

        # If no windows were kept from this segment (possibly due to filtering), count it as a dropped segment if it was a gesture, and skip adding it to the dataset.
        if len(y_segment) == 0:
            if expected_label != "Rest":
                dropped_segment_count += 1
            continue
        
        kept_window_count += len(y_segment)

        # Append the windows and labels from this segment to the overall dataset lists, along with file and subject information for later analysis.
        X_list.append(X_segment)
        y_list.append(y_segment)
        file_list.extend([segment_df["source_file"].iloc[0]] * len(y_segment))
        subject_list.extend([segment_df["subject_id"].iloc[0]] * len(y_segment))

        segment_ids.extend([segment_id] * len(y_segment))

    if len(X_list) == 0:
        raise RuntimeError(
            "No training windows were created. Try reducing window_size, "
            "reducing step_size, lowering min_segment_seconds, or disabling the Rest-energy filter."
        )
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    files = np.array(file_list)
    subjects = np.array(subject_list)
    segment_ids = np.array(segment_ids)

    # <insert splitting code>
    train_groups, val_groups = split_by_group_class_aware(
        groups=segment_ids,
        labels=y,
        train_size=0.8,
        val_size=0.2,
        random_state=42,
        max_tries=1000
    )

    train_mask = np.array([g in train_groups for g in segment_ids])
    val_mask = np.array([g in val_groups for g in segment_ids])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    #encode labels
    y_train_encoded = encode_labels(y_train, label_map)
    y_val_encoded = encode_labels(y_val, label_map)

    #normalize features
    X_train_fe = prepare_model_features(X_train)
    X_val_fe = prepare_model_features(X_val)

    mean_std = fit_channel_standardizer(X_train_fe)

    X_train_n = apply_channel_standardization(X_train_fe, *mean_std)
    X_val_n = apply_channel_standardization(X_val_fe, *mean_std)

    num_classes = len(label_map)

    #create data loaders
    batch_size = 64

    train_dataset = EMGWindowsDataset(
        X_train_n,
        y_train_encoded,
        augment=True,
        noise_std=0.005,
        gain_jitter_std=0.02
    )

    val_dataset = EMGWindowsDataset(
        X_val_n,
        y_val_encoded,
        augment=False
    )

    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)

    class_counts = torch.bincount(
        y_train_tensor,
        minlength=num_classes
    ).float()

    class_counts = torch.clamp(class_counts, min=1.0)

    class_sample_weights = 1.0 / class_counts
    sample_weights = class_sample_weights[y_train_tensor]

    train_sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader

#training and evaluation functions
def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device) -> dict:
    """
    Train the model for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count
    }

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device) -> dict:
    """
    Evaluate the model and return loss, accuracy, and macro-F1.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    all_preds = []
    all_labels = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == y_batch).sum().item()
        total_count += batch_size

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "macro_f1": macro_f1
    }

def _train(model, sig):
    """
    Trains a model in place using MindRove signal and the associated labels
    Parameters:
        model: Torch model (CNNLSTMClassifier)
        sig:   NumPy array (3D: 1 batch, 200 rows, 8 columns)
        label: NumPy array (1D: 200 elements)
    Returns:
        Nothing
    """
    print("ModelManager.train received:")

    print(sig.shape)
    print(sig)
    #create loaders
    train_loader, val_loader = prepare_loaders(sig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set requires_grad for all parameters to True, then freeze CNN layers
    for param in model.parameters():
        param.requires_grad = True

    for param in model.cnn.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(
    label_smoothing=0.0
)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=5e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6
    )
  
    max_epochs = 50
    patience = 15
    best_val_macro_f1 = -math.inf
    best_model_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device) if len(val_loader) > 0 else None

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"]
        }

        if val_metrics is not None:
            row["val_loss"] = val_metrics["loss"]
            row["val_accuracy"] = val_metrics["accuracy"]
            row["val_macro_f1"] = val_metrics["macro_f1"]

            scheduler.step(val_metrics["macro_f1"])

            if val_metrics["macro_f1"] > best_val_macro_f1 + 1e-4:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['accuracy']:.4f}, "
                f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}, "
                f"Val MacroF1={val_metrics['macro_f1']:.4f}, "
                f"NoImprove={epochs_no_improve}/{patience}"
            )

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no macro-F1 improvement")
                break

        else:
            print(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['accuracy']:.4f}"
            )

        history.append(row)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded Best Model With Best Val Macro F1={best_val_macro_f1:.4f}")
