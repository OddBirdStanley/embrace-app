import torch
import torch.nn as nn
from collections import deque
import gc
import os
import time
import numpy as np
from threading import Lock
BIN_PATH = os.path.join(os.path.dirname(__file__), "bin")

from PySide6.QtCore import QThread, Signal, Slot

class CNNLSTMClassifier(nn.Module):
    MEAN = np.array([
        -2.2562135200132616e-05,
        -2.056811354123056e-05,
        -4.008084943052381e-06,
        3.388740515219979e-05,
        -9.689700164017268e-06,
        9.724450501380488e-05,
        -1.0648847819538787e-05,
        -7.948076017783023e-06,
        0.4911661148071289,
        -0.011800535023212433,
        -0.08037097007036209,
        -0.06681600958108902,
        -0.09873779118061066,
        -0.12540459632873535,
        -0.20253531634807587,
        -0.006005008239299059
    ])
    STD = np.array([
        939.9639282226562,
        64.4369125366211,
        70.32238006591797,
        102.73607635498047,
        168.22476196289062,
        190.44500732421875,
        230.51136779785156,
        133.44361877441406,
        65.2059326171875,
        62.25019454956055,
        72.54779815673828,
        77.90699768066406,
        102.0621109008789,
        125.46758270263672,
        35.52862548828125,
        34.05629348754883
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
        return logits

MODEL_CONFIG = {
    "SuperTony": {
        "clazz": CNNLSTMClassifier,
        "weights": "best_model_v10.pt",
        "window": 200,
        "step": 25
    }
}

class ModelManager:
    def __init__(self):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
    
    def set_model(self, name):
        self.config = MODEL_CONFIG[name]
        self.model = self.config["clazz"]().to(self.dev)
        gc.collect()
        torch.cuda.empty_cache()
        self.model.load_state_dict(torch.load(os.path.join(BIN_PATH, self.config["weights"])))
    
    def predict(self, sig):
        return self.model(torch.tensor(self.model.pre(sig)).to(self.dev)).cpu().detach().numpy()

class ModelThread(QThread):
    deposit = Signal(object)
    predicted = Signal(int)
    stop = Signal()

    def __init__(self, manager):
        super().__init__()

        self.manager = manager
        self.alive = True
        self.q = deque()
        self.lock = Lock()

        self.deposit.connect(self.handle_deposit)
        self.stop.connect(self.cleanup)
    
    def handle_deposit(self, data):
        self.lock.acquire()
        for i in range(data.shape[0]):
            self.q.append(data[i].copy())
        self.lock.release()
    
    def cleanup(self):
        self.lock.acquire()
        self.alive = False
        self.lock.release()
    
    def run(self):
        while True:
            self.lock.acquire()
            if not self.alive:
                self.lock.release()
                break
            samples = None
            if len(self.q) >= self.manager.config["window"]:
                samples = []
                for i in range(self.manager.config["window"]):
                    samples.append(self.q.popleft())
                for i in range(len(samples) - 1, self.manager.config["step"] - 1, -1):
                    self.q.appendleft(samples[i].copy())
            self.lock.release()
            if samples is not None:
                samples = np.vstack(samples, dtype=np.float32)
                samples = samples[np.newaxis, :]
                self.predicted.emit(int(np.argmax(self.manager.predict(samples))))
            time.sleep(0.1)

    