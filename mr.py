import subprocess
import time
from threading import Lock
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot
from openmr.stream import MindRoveStream
from openmr.board_metadata import get_emg_channels

CONNECT_FAILURE = 0
CONNECT_SUCCESS = 1
CONNECT_NORMAL = 2

class MindRoveConnection(QThread):
    connected = Signal(int)
    cleanup_complete = Signal(int)
    update = Signal(object)
    stop = Signal()

    def __init__(self):
        super().__init__()

        self.stop.connect(self.cleanup)
        self.stream = None
        self.alive = True
        self.has_error = False
        self.lock = Lock()
        self.destroyed.connect(self.cleanup)
    
    @Slot()
    def cleanup(self):
        try:
            self.stream.stop()
        except:
            pass
        self.lock.acquire()
        self.alive = False
        self.cleanup_complete.emit(CONNECT_FAILURE if self.has_error else CONNECT_NORMAL)
        self.lock.release()

    def run(self):
        try:
            self.stream = MindRoveStream()
            self._channels = get_emg_channels()
            self.stream.start()
        except:
            self.has_error = True
            self.connected.emit(CONNECT_FAILURE)
            return
        self.connected.emit(CONNECT_SUCCESS)

        while True:
            self.lock.acquire()
            if not self.alive:
                self.lock.release()
                break
            self.lock.release()
            try:
                data = self.stream.get_data()[self._channels, :].transpose()
                if data.shape[0] > 0:
                    self.update.emit(data)
            except:
                self.lock.acquire()
                self.has_error = True
                self.lock.release()
                break
            time.sleep(0.1)

        self.lock.acquire()
        self.connected.emit(CONNECT_FAILURE if self.has_error else CONNECT_NORMAL)
        self.lock.release()