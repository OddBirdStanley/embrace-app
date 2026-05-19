import subprocess
import time
from threading import Lock
import numpy as np
from PySide6.QtCore import QThread, Signal, Slot
from random import randint
from openmr.stream import MindRoveStream
from openmr.board_metadata import get_emg_channels

CONNECT_FAILURE = 0
CONNECT_SUCCESS = 1
CONNECT_NORMAL = 2

class MindRoveRecord(QThread):
    instruction = Signal(object)
    end = Signal()
    stop = Signal()

    def __init__(self, type_count, interval=9):
        super().__init__()

        self.type_count = type_count
        self.interval = interval
        self.alive = True
        self.lock = Lock()
        self.stop.connect(self.cleanup)
        self.destroyed.connect(self.cleanup)
    
    def cleanup(self):
        self.lock.acquire()
        self.alive = False
        self.lock.release()
    
    def run(self):
        index = 0
        for i in range(3, 0, -1):
            self.instruction.emit(("cd", i, index))
            self.lock.acquire()
            if not self.alive:
                self.end.emit()
                self.lock.release()
                return
            self.lock.release()
            time.sleep(1)

        while True:
            index_next = (index + 1) % self.type_count
            for i in range(self.interval, 0, -1):
                self.lock.acquire()
                if not self.alive:
                    self.lock.release()
                    break
                self.lock.release()
                self.instruction.emit(("use", index, index_next, i))
                time.sleep(1)
            self.lock.acquire()
            if not self.alive:
                self.lock.release()
                break
            self.lock.release()
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
        self.lock.acquire()
        self.alive = False
        self.cleanup_complete.emit(CONNECT_FAILURE if self.has_error else CONNECT_NORMAL)
        self.lock.release()

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