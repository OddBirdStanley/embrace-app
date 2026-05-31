import bleak
import asyncio
import time
from collections import deque
from PySide6.QtCore import QThread, Signal, Slot
from threading import Lock

SERVICE = ["bf77b1ec-80a9-4849-9c64-e6dcd32eb5c9"]
CHARACTERISTIC = "49b7f353-b294-4023-abb7-1976b9494c2e"

TEST = bytearray(0x70)

CONNECT_SUCCESS = 0
CONNECT_FAILURE = 1
CONNECT_NORMAL = 2

class BLEDiscover(QThread):
    complete = Signal(object)

    async def discover(self):
        try:
            devices = await bleak.BleakScanner().discover(timeout=5, service_uuids=SERVICE)
            return devices
        except:
            pass
        return None

    def run(self):
        self.complete.emit(asyncio.run(self.discover()))

class BLEConnection(QThread):
    connected = Signal(int)
    cleanup_complete = Signal(int)
    deposit = Signal(int)
    stop = Signal()

    def __init__(self, address):
        super().__init__()

        self.stop.connect(self.cleanup)
        self.address = address
        self.client = None
        self.alive = True
        self.has_error = False
        self.q = deque()
        self.lock = Lock()
        self.destroyed.connect(self.cleanup)
        self.deposit.connect(self.handle)
    
    @Slot(int)
    def handle(self, i):
        with self.lock:
            self.q.append(i)
    
    def cleanup(self):
        try:
            asyncio.run(self.client.disconnect())
        except:
            pass
        with self.lock:
            self.alive = False
            self.cleanup_complete.emit(CONNECT_FAILURE if self.has_error else CONNECT_NORMAL)

    def run(self):
        # connect to BLE
        self.client = bleak.BleakClient(self.address)
        try:
            asyncio.run(self.client.connect(timeout=5))
        except:
            self.has_error = True
            print("err source 1")
            self.connected.emit(CONNECT_FAILURE)
            return

        with self.lock:
            if not self.alive:
                self.has_error = True
                print("err source 2")
                self.connected.emit(CONNECT_FAILURE)
                return # destroyed auto-cleanup
        self.connected.emit(CONNECT_SUCCESS)

        while True:
            with self.lock:
                if not self.alive:
                    break
                while len(self.q) > 0:
                    self.send(bytearray([self.q.popleft()]))
            time.sleep(0.01)
        
        with self.lock:
            self.connected.emit(CONNECT_FAILURE if self.has_error else CONNECT_NORMAL)
    
    async def _send(self, data):
        try:
            await self.client.write_gatt_char(
                CHARACTERISTIC,
                data
            )
        except:
            return False
        return True

    def send(self, data):
        if not asyncio.run(self._send(data)):
            print("err source 3")
            self.alive = False
            self.has_error = True
