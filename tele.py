from flask import Flask
from PySide6.QtCore import QThread, Signal, Slot
from werkzeug.serving import make_server

app = Flask("embrace-app.tele")

@app.route("/")
def home():
    return "hello world!"

class TeleThread(QThread):
    stop = Signal()

    def __init__(self):
        super().__init__()
        self.server = make_server("127.0.0.1", 8000, app)
        self.stop.connect(self.handle_signal)

    def run(self):
        self.server.serve_forever()
    
    def handle_signal(self):
        self.server.shutdown()