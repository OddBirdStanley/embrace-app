import bleak
import asyncio
import time
from PySide6.QtCore import QThread, Signal
from threading import Lock

SERVICE = None
CHARACTERISTIC = None

TEST = bytearray(0x7f)

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
    connected = Signal(bool)

    def __init__(self, address):
        super().__init__()

        self.address = address
        self.client = None
        self.alive = True
        self.lock = Lock()
        self.destroyed.connect(self.cleanup)
    
    def stop(self):
        self.alive = False
    
    def cleanup(self):
        try:
            asyncio.run(self.client.disconnect())
        except:
            pass
        print("Bluetooth cleanup successful")

    def run(self):
        self.client = bleak.BleakClient(self.address)
        try:
            asyncio.run(self.client.connect(timeout=5))
        except:
            self.connected.emit(False)
            return
        self.send(TEST)
        self.lock.acquire()
        if not self.alive:
            return
        self.connected.emit(True)
        self.lock.release()

        while True:
            self.lock.acquire()
            if not self.alive:
                break
            self.lock.release()
            time.sleep(1)
        
        self.connected.emit(False)
    
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
        self.lock.acquire()
        if not asyncio.run(self._send(data)):
            self._alive = False
        self.lock.release()