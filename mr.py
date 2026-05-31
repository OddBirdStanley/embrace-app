import subprocess
import time
import os
from threading import Lock
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot
from random import randint
from openmr.stream import MindRoveStream
from openmr.board_metadata import get_emg_channels

CONNECT_FAILURE = 0
CONNECT_SUCCESS = 1
CONNECT_NORMAL = 2

DEBUG_MRFZ = os.getenv("EMBRACE_MRFZ") == "1"

class MindRoveRecord(QThread):
    instruction = Signal(object)
    end = Signal()
    stop = Signal()

    def __init__(self, type_count, limit, interval=3):
        super().__init__()

        self.type_count = type_count
        self.limit = type_count * 2 if limit else -1
        self.interval = interval
        self.alive = True
        self.lock = Lock()
        self.stop.connect(self.cleanup)
        self.destroyed.connect(self.cleanup)
    
    def cleanup(self):
        with self.lock:
            self.alive = False
    
    def run(self):
        index = randint(1, self.type_count)
        for i in range(self.interval, 0, -1):
            self.instruction.emit(("cd", i, index - 1))
            with self.lock:
                if not self.alive:
                    self.end.emit()
                    return
            time.sleep(1)

        while self.limit != 0:
            self.limit -= 1
            index_next = -index
            if index_next > 0:
                index_next += 1
                if index_next > self.type_count:
                    index_next = 1
            for i in range(self.interval, 0, -1):
                with self.lock:
                    if not self.alive:
                        break
                self.instruction.emit(("use", index - 1, index_next - 1, i))
                time.sleep(1)
            with self.lock:
                if not self.alive:
                    break
            index = index_next
        
        self.end.emit()

class MindRoveConnection(QThread):
    connected = Signal(int)
    cleanup_complete = Signal(int)
    update = Signal(object)
    stop = Signal()

    def __init__(self):
        super().__init__()

        self.stream = None
        self.alive = True
        self.has_error = False
        self.lock = Lock()
        self.destroyed.connect(self.cleanup)
        self.stop.connect(self.cleanup)

    def cleanup(self):
        try:
            self.stream.stop()
        except:
            pass
        with self.lock:
            self.alive = False
            self.cleanup_complete.emit(CONNECT_FAILURE if self.has_error else CONNECT_NORMAL)

    def run(self):
        try:
            self.stream = MindRoveStream()
            self._channels = get_emg_channels()
            self.stream.start(-1)
        except:
            self.has_error = True
            self.connected.emit(CONNECT_FAILURE)
            return
        self.connected.emit(CONNECT_SUCCESS)

        while True:
            with self.lock:
                if not self.alive:
                    break
            try:
                data = self.stream.get_data()[self._channels, :].transpose()
                if data.shape[0] > 0:
                    self.update.emit(data)
                elif DEBUG_MRFZ:
                    self.update.emit(np.zeros((randint(50, 150), 8)))
            except:
                with self.lock:
                    self.has_error = True
                break
            time.sleep(0.01)

        with self.lock:
            self.connected.emit(CONNECT_FAILURE if self.has_error else CONNECT_NORMAL)