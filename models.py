import torch
import torch.nn as nn
from collections import deque
import gc
import os
import json
import time
import numpy as np
from threading import Lock
BIN_PATH = os.path.join(os.path.dirname(__file__), "bin")

from PySide6.QtCore import QThread, Signal, Slot

class CNNLSTMClassifier(nn.Module):
    MEAN = np.array([
        -2.0922023395542055e-05,
        -1.5901439837762155e-05,
        1.3434827451419551e-05,
        -4.066952442371985e-06,
        2.035843863268383e-05,
        5.7909997849492356e-05,
        -1.6111873264890164e-05,
        -1.9541965230018832e-05,
        0.7620308995246887,
        0.5067156553268433,
        0.5156113505363464,
        0.5203160643577576,
        0.5006045699119568,
        0.5636326670646667,
        0.6903709769248962,
        0.6672220230102539
    ])
    STD = np.array([
        871.6812744140625,
        651.9368896484375,
        655.3069458007812,
        657.824951171875,
        659.859619140625,
        647.8235473632812,
        812.7166137695312,
        663.9458618164062,
        54.2520866394043,
        53.20650100708008,
        61.22278594970703,
        68.18769073486328,
        90.08158111572266,
        102.6977310180664,
        33.157127380371094,
        31.934154510498047
    ])

    def __init__(
        self,
        input_size: int = 16,
        num_classes: int = 8,
        conv_channels: list[int] | tuple[int, ...] = (64, 128),
        kernel_size: int = 7,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        dropout: float = 0.15,
        bidirectional: bool = True,
    ):
        """
        Initiate the CNNLSTMClassifier with the specified architecture parameters, including the input size (number of channels), number of output classes, convolutional layer configuration, LSTM configuration, dropout rate, and whether to use bidirectional LSTMs.
        """
        
        # Call the superclass constructor to initialize the nn.Module
        super().__init__()

        # Store the parameters as instance variables
        self.input_size = input_size
        self.num_classes = num_classes
        self.conv_channels = list(conv_channels)
        self.kernel_size = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # CNN expects shape: (batch, channels, time)
        conv_layers = []
        in_channels = input_size

        # Build the convolutional layers based on the specified configuration
        for i, out_channels in enumerate(self.conv_channels):
            # Each convolutional layer consists of a Conv1d, followed by BatchNorm1d, ReLU activation, and MaxPool1d to reduce the temporal dimension. 
            conv_layers.extend([
                nn.Conv1d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    padding = kernel_size // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ])

            # Only pool once, after the first conv block
            if i == 0:
                conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

            # Update in_channels for the next layer to be the out_channels of the current layer
            in_channels = out_channels
        
        # Combine the convolutional layers into a sequential module
        self.cnn = nn.Sequential(*conv_layers)

        # LSTM input size becomes the final number of CNN output channels
        self.lstm = nn.LSTM(
            input_size = in_channels,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers,
            batch_first = True,
            dropout = dropout if lstm_num_layers > 1 else 0.0,
            bidirectional = bidirectional
        )

        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)

        # avg pool + max pool => 2 * lstm_output_size
        self.head = nn.Sequential(
            nn.Linear(lstm_output_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def remove_window_dc_offset(self, X):
        window_mean = np.mean(X, axis=1, keepdims=True)
        return (X - window_mean).astype(np.float32)
    def add_delta_features(self, X):
        delta = np.diff(X, axis=1, prepend=X[:, :1, :])
        return np.concatenate([X, delta], axis=2).astype(np.float32)
    def prepare_model_features(self, X):
        X_centered = self.remove_window_dc_offset(X)
        X_features = self.add_delta_features(X_centered)
        return X_features.astype(np.float32)
    def apply_channel_standardization(self, X):
        return ((X - self.MEAN[None, None, :]) / self.STD[None, None, :]).astype(np.float32)
    def pre(self, X):
        return self.apply_channel_standardization(self.prepare_model_features(X))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, num_channels)

        # CNN expects (batch, channels, time)
        x = x.transpose(1, 2)

        # CNN feature extraction
        x = self.cnn(x)

        # Back to LSTM shape: (batch, seq_len, features)
        x = x.transpose(1, 2)

        # LSTM output for every time step
        lstm_out, _ = self.lstm(x)

        # Global average pooling over time
        avg_pool = torch.mean(lstm_out, dim=1)

        # Global max pooling over time
        max_pool, _ = torch.max(lstm_out, dim=1)

        # Combine both summaries
        features = torch.cat([avg_pool, max_pool], dim=1)

        logits = self.head(features)
        return torch.softmax(logits, dim=1)

with open(os.path.join(BIN_PATH, "model_config.json")) as f:
    MODEL_CONFIG = json.loads(f.read())
MODEL_CLASS_MAPPING = {"cnn-lstm": CNNLSTMClassifier}

from plug import _train
class ModelManager:
    def __init__(self):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
    
    def set_model(self, name):
        self.config = MODEL_CONFIG[name]
        self.model = MODEL_CLASS_MAPPING[self.config["clazz"]]().to(self.dev)
        gc.collect()
        torch.cuda.empty_cache()
        self.model.load_state_dict(torch.load(os.path.join(BIN_PATH, self.config["weights"]), map_location=self.dev))
    
    def predict(self, sig):
        return self.model(torch.tensor(self.model.pre(sig)).to(self.dev)).cpu().detach().numpy()
    
    def train(self, arr):
        _train(self.model, arr)

class ModelThread(QThread):
    deposit = Signal(object)
    predicted = Signal(object)
    stop = Signal()

    fine_tune = Signal(object)
    fine_tune_end = Signal()

    def __init__(self, manager):
        super().__init__()

        self.manager = manager
        self.alive = True
        self.q = deque()
        self.lock = Lock()
        self.deposit.connect(self.handle_deposit)
        self.stop.connect(self.cleanup)
        self.fine_tune.connect(self.handle_fine_tune)

        self.fine_tune_data = None
    
    @Slot(object)
    def handle_fine_tune(self, arr):
        with self.lock:
            self.fine_tune_data = arr
    
    def set_model(self, name):
        with self.lock:
            self.manager.set_model(name)
    
    def handle_deposit(self, data):
        with self.lock:
            for i in range(data.shape[0]):
                self.q.append(data[i].copy())

    def cleanup(self):
        with self.lock:
            self.alive = False
    
    def run(self):
        while True:
            with self.lock:
                if not self.alive:
                    break
                
                if self.fine_tune_data is not None:
                    self.manager.train(self.fine_tune_data)
                    self.fine_tune_data = None
                    self.fine_tune_end.emit()
                
                samples = None
                if len(self.q) >= self.manager.config["window"]:
                    samples = []
                    for i in range(self.manager.config["window"]):
                        samples.append(self.q.popleft())
                    for i in range(len(samples) - 1, self.manager.config["step"] - 1, -1):
                        self.q.appendleft(samples[i].copy())
            
            if samples is not None:
                _samples = np.vstack(samples, dtype=np.float32)[np.newaxis, :]
                with self.lock:
                    probs = self.manager.predict(_samples)
                    index = int(np.argmax(probs))
                    self.predicted.emit((index, probs[0][index]))
            time.sleep(0.1)
    