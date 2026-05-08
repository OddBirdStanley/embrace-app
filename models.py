import torch
import torch.nn as nn
from collections import deque
import os
import time
import numpy as np
from threading import Lock
BIN_PATH = os.path.join(os.path.dirname(__file__), "bin")

from PySide6.QtCore import QThread, Signal, Slot

class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        num_classes: int = 6,
        conv_channels: list[int] | tuple[int, ...] = (48, 96),
        kernel_size: int = 7,
        lstm_hidden_size: int = 96,
        lstm_num_layers: int = 1,
        dropout: float = 0.25,
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
    "Tony": {
        "clazz": CNNLSTMClassifier,
        "weights": "best_model_v8.pt"
    }
}

class ModelManager:
    def __init__(self):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
    
    def set_model(self, name):
        self.model = MODEL_CONFIG[name]["clazz"]().to(self.dev)
        self.model.load_state_dict(torch.load(os.path.join(BIN_PATH, MODEL_CONFIG[name]["weights"])))
    
    def predict(self, sig):
        return self.model(torch.tensor(sig).to(self.dev)).cpu().detach().numpy()

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
                break
            samples = None
            if len(self.q) >= 600:
                samples = []
                for i in range(600):
                    samples.append(self.q.popleft())
            self.lock.release()
            if samples is not None:
                samples = np.vstack(samples, dtype=np.float32)
                samples = samples[np.newaxis, :]
                self.predicted.emit(int(np.argmax(self.manager.predict(samples))))
            time.sleep(0.1)

    