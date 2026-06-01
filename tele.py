from flask import Flask, send_from_directory, request, Response
from PySide6.QtCore import QThread, Signal, Slot
from werkzeug.serving import make_server
import os

ROOT_PATH = os.path.dirname(__file__)

class TeleThread(QThread):
    stop = Signal()

    def __init__(self, app):
        super().__init__()
        self._flask = Flask("embrace-app.tele")
        self.app = app

        @self._flask.route("/")
        def home():
            return send_from_directory(ROOT_PATH, "index.html")

        @self._flask.route("/control")
        def control():
            code = request.args.get("code")
            code = int(code)
            if 0 <= code < 8:
                self.app.pred_sims[code].click()
            return Response(status=200)
        
        self.server = make_server("0.0.0.0", 8000, self._flask)
        self.stop.connect(self.handle_signal)

    def run(self):
        self.server.serve_forever()
    
    def handle_signal(self):
        self.server.shutdown()