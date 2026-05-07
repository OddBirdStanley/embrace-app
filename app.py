from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Signal, Slot

import ble
import models
import styles

def make_sep():
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setLineWidth(2)
    sep.setStyleSheet("color: grey;")
    return sep

class EmbraceState:
    def __init__(self):
        self.mindrove = None
        self.arm = None
        self.model_manager = models.ModelManager()
        self.ble_connection = None

class ArmDialog(QDialog):
    complete = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self.devices = []
        self.setWindowTitle("Bluetooth Arms")
        self.setWindowModality(Qt.WindowModal)

        root_layout = QVBoxLayout()
        self.discovered = QComboBox()
        self.discovered.setMinimumWidth(300)
        self.button_yes = QPushButton("Connect")
        self.button_refresh = QPushButton("Refresh")
        self.button_yes.clicked.connect(self.yes)
        self.button_refresh.clicked.connect(self.refresh)
        buttons = QHBoxLayout()
        root_layout.addWidget(self.discovered)
        buttons.addWidget(self.button_refresh)
        buttons.addWidget(self.button_yes)
        root_layout.addLayout(buttons)
        self.setLayout(root_layout)
        
        self.refresh()
    
    def refresh(self):
        self.devices = []
        self.discovered.clear()
        self.discovered.setEnabled(False)
        self.button_yes.setEnabled(False)
        self.button_refresh.setEnabled(False)
        self._thread = ble.BLEDiscover()
        self._thread.complete.connect(self.refresh_complete)
        self._thread.start()
    
    def yes(self):
        self.complete.emit(self.devices[self.discovered.currentIndex()])
        self.close()
    
    @Slot(object)
    def refresh_complete(self, devices):
        if devices is None:
            error = QMessageBox(self)
            error.setIcon(QMessageBox.Icon.Warning)
            error.setWindowTitle("Error")
            error.setText("Bluetooth is not available.")
            error.setStandardButtons(QMessageBox.StandardButton.Ok)
            error.finished.connect(self.close)
            error.show()
        else:
            self.devices = devices
            self.discovered.addItems([f"{i.name} ({i.address})" for i in self.devices])
            self.discovered.setEnabled(True)
            if self.devices:
                self.button_yes.setEnabled(True)
            self.button_refresh.setEnabled(True)

class EmbraceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.state = EmbraceState()

        root_layout = QVBoxLayout(self)
        control_layout = QHBoxLayout()
        mindrove_status = QVBoxLayout()
        mindrove_status_label_1 = QLabel("MindRove")
        mindrove_status_label_2 = QLabel("not connected")
        mindrove_status_label_2.setStyleSheet(styles.LABEL_NO)
        mindrove_status.addWidget(mindrove_status_label_1)
        mindrove_status.addWidget(mindrove_status_label_2)
        mindrove_status_label_1.setAlignment(Qt.AlignCenter)
        mindrove_status_label_2.setAlignment(Qt.AlignCenter)
        control_layout.addLayout(mindrove_status)
        mindrove_connect = QPushButton("Connect")
        control_layout.addWidget(mindrove_connect)
        arm_status = QVBoxLayout()
        arm_status_label_1 = QLabel("Arm")
        self.arm_status_label_2 = QLabel("not connected")
        self.arm_status_label_2.setStyleSheet(styles.LABEL_NO)
        arm_status.addWidget(arm_status_label_1)
        arm_status.addWidget(self.arm_status_label_2)
        arm_status_label_1.setAlignment(Qt.AlignCenter)
        self.arm_status_label_2.setAlignment(Qt.AlignCenter)
        control_layout.addLayout(arm_status)
        self.arm_connect = QPushButton("Connect")
        self.arm_connect_handle = self.arm_connect.clicked.connect(self.arm_dialog_show)
        control_layout.addWidget(self.arm_connect)
        model_status = QVBoxLayout()
        model_choose = QHBoxLayout()
        model_choose_label = QLabel("Model:")
        model_choose_menu = QComboBox()
        model_choose_menu.addItems(models.MODEL_CONFIG.keys())
        self.state.model_manager.set_model(model_choose_menu.currentText())
        model_choose.addWidget(model_choose_label)
        model_choose.addWidget(model_choose_menu)
        model_status.addLayout(model_choose)
        model_status_cuda = QLabel(f"CUDA: {'OK' if self.state.model_manager.dev == 'cuda' else 'none'}")
        model_status_cuda.setStyleSheet(styles.LABEL_YES if self.state.model_manager.dev == "cuda" else styles.LABEL_NO)
        model_status_cuda.setAlignment(Qt.AlignCenter)
        model_status.addWidget(model_status_cuda)
        control_layout.addLayout(model_status)

        sigs = [QProgressBar() for i in range(8)]
        sig_layout = QVBoxLayout()
        for i in range(8):
            sigs[i].setTextVisible(True)
            sigs[i].setValue(0)
            sigs[i].setFormat(f"Channel {i + 1}")
            sigs[i].setEnabled(False)
            sig_layout.addWidget(sigs[i])

        preds = [
            QLabel("Extend"),
            QLabel("Fist"),
            QLabel("Flex"),
            QLabel("Radial"),
            QLabel("Rest"),
            QLabel("Ulnar")
        ]
        pred_layout = QHBoxLayout()
        for l in preds:
            pred_layout.addWidget(l)
            l.setFixedWidth(100)
            l.setFixedHeight(100)
            l.setAlignment(Qt.AlignCenter)
            l.setStyleSheet(styles.PRED_INACTIVE)

        root_layout.addLayout(control_layout)
        root_layout.addWidget(make_sep())
        sig_label = QLabel("Signals")
        sig_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        pred_label = QLabel("Predictions")
        pred_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        root_layout.addWidget(sig_label)
        root_layout.addLayout(sig_layout)
        root_layout.addWidget(make_sep())
        root_layout.addWidget(pred_label)
        root_layout.addLayout(pred_layout)
    
    def arm_dialog_show(self):
        dialog = ArmDialog(self)
        dialog.complete.connect(self.arm_dialog_return)
        dialog.show()
    
    @Slot(object)
    def arm_dialog_return(self, device):
        self.arm_connect.setEnabled(False)
        self.state.ble_connection = ble.BLEConnection(device.address)
        self.destroyed.connect(self.state.ble_connection.cleanup)
        self.state.ble_connection.connected.connect(self.arm_dialog_status)
        self.state.ble_connection.start()
    
    @Slot(bool)
    def arm_dialog_status(self, status):
        if status:
            self.arm_status_label_2.setText("connected")
            self.arm_status_label_2.setStyleSheet(styles.LABEL_YES)
            self.arm_connect.setText("Disconnect")
            self.arm_connect.disconnect(self.arm_connect_handle)
            self.arm_connect_handle = self.arm_connect.clicked.connect(self.state.ble_connection.stop)
            self.arm_connect.setEnabled(True)
        else:
            self.arm_status_label_2.setText("not connected")
            self.arm_status_label_2.setStyleSheet(styles.LABEL_NO)
            self.state.ble_connection.cleanup()
            self.state.ble_connection = None
            self.arm_connect.disconnect(self.arm_connect_handle)
            self.arm_connect_handle = self.arm_connect.clicked.connect(self.arm_dialog_show)
            self.arm_connect.setText("Connect")
            self.arm_connect.setEnabled(True)
            error = QMessageBox(self)
            error.setIcon(QMessageBox.Icon.Warning)
            error.setWindowTitle("Error")
            error.setText("Bluetooth connection failed.")
            error.setStandardButtons(QMessageBox.StandardButton.Ok)
            error.show()

if __name__ == "__main__":
    app = QApplication([])
    app.setApplicationDisplayName("Embrace")
    window = EmbraceApp()
    window.show()
    app.exec()