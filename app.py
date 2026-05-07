from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Signal, Slot

import ble
import mr
import models
import styles

class EmbraceState:
    def __init__(self):
        self.mindrove = None
        self.arm = None
        self.model_manager = models.ModelManager()
        self.ble_connection = None
        self.mr_connection = None

class DeviceDiscoveryDialog(QDialog):
    complete = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self.devices = []
        self.setWindowTitle(self.title())
        self.setWindowModality(Qt.WindowModal)

        root_layout = QVBoxLayout()
        self.discovered = QComboBox()
        self.discovered.setMinimumWidth(300)
        self.button_yes = QPushButton("Connect")
        self.button_refresh = QPushButton("Refresh")
        self.button_yes.clicked.connect(self._yes)
        self.button_refresh.clicked.connect(self._refresh)
        buttons = QHBoxLayout()
        root_layout.addWidget(self.discovered)
        buttons.addWidget(self.button_refresh)
        buttons.addWidget(self.button_yes)
        root_layout.addLayout(buttons)
        self.setLayout(root_layout)
        
        self._refresh()
    
    def _refresh(self):
        self.devices = []
        self.discovered.clear()
        self.discovered.setEnabled(False)
        self.button_yes.setEnabled(False)
        self.button_refresh.setEnabled(False)
        self._thread = self.thread()
        self._thread.complete.connect(self._refresh_complete)
        self.destroyed.connect(self._thread.terminate)
        self._thread.start()
    
    def _yes(self):
        self.complete.emit(self.return_value())
        self.close()
    
    @Slot(object)
    def _refresh_complete(self, devices):
        if devices is None:
            error = QMessageBox(self)
            error.setIcon(QMessageBox.Icon.Warning)
            error.setWindowTitle("Error")
            error.setText(self.error_message())
            error.setStandardButtons(QMessageBox.StandardButton.Ok)
            error.finished.connect(self.close)
            error.show()
        else:
            self.devices = devices
            self.discovered.addItems([self.item_label(i) for i in self.devices])
            self.discovered.setEnabled(True)
            if self.devices:
                self.button_yes.setEnabled(True)
            self.button_refresh.setEnabled(True)

    # Abstract methods
    
    def title(self):
        raise NotImplementedError
    
    def thread(self):
        raise NotImplementedError
    
    def return_value(self):
        raise NotImplementedError
    
    def error_message(self):
        raise NotImplementedError
    
    def item_label(self, item):
        raise NotImplementedError

class ArmDialog(DeviceDiscoveryDialog):
    def __init__(self, parent):
        super().__init__(parent)
    
    def title(self):
        return "Bluetooth Arms"
    
    def thread(self):
        return ble.BLEDiscover()
    
    def return_value(self):
        return self.devices[self.discovered.currentIndex()].address
    
    def error_message(self):
        return "Bluetooth is not available."
    
    def item_label(self, item):
        return f"{item.name} ({item.address})"

class EmbraceApp(QWidget):
    MAX_SIG = 1e5
    MIN_SIG = -1e5

    def __init__(self):
        super().__init__()
        self.state = EmbraceState()
        self.model_thread = models.ModelThread(self.state.model_manager)
        self.model_thread.predicted.connect(self.model_callback)
        self.model_thread.start()

        root_layout = QVBoxLayout(self)
        control_layout = QHBoxLayout()
        mindrove_status = QVBoxLayout()
        mindrove_status_label_1 = QLabel("MindRove")
        self.mindrove_status_label_2 = QLabel("not connected")
        self.mindrove_status_label_2.setStyleSheet(styles.LABEL_NO)
        mindrove_status.addWidget(mindrove_status_label_1)
        mindrove_status.addWidget(self.mindrove_status_label_2)
        mindrove_status_label_1.setAlignment(Qt.AlignCenter)
        self.mindrove_status_label_2.setAlignment(Qt.AlignCenter)
        control_layout.addLayout(mindrove_status)
        self.mindrove_connect = QPushButton("Connect")
        self.mindrove_connect_handle = self.mindrove_connect.clicked.connect(self.mindrove_connection_start)
        control_layout.addWidget(self.mindrove_connect)
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

        self.sigs = [QProgressBar() for i in range(8)]
        sig_layout = QVBoxLayout()
        for i in range(8):
            self.sigs[i].setTextVisible(True)
            self.sigs[i].setValue(0)
            self.sigs[i].setMinimum(0)
            self.sigs[i].setMaximum(1)
            self.sigs[i].setFormat(f"Channel {i + 1}")
            self.sigs[i].setEnabled(False)
            sig_layout.addWidget(self.sigs[i])

        self.preds = [
            QLabel("Extend"),
            QLabel("Fist"),
            QLabel("Flex"),
            QLabel("Radial"),
            QLabel("Rest"),
            QLabel("Ulnar")
        ]
        pred_layout = QHBoxLayout()
        for l in self.preds:
            pred_layout.addWidget(l)
            l.setFixedWidth(100)
            l.setFixedHeight(100)
            l.setAlignment(Qt.AlignCenter)
            l.setStyleSheet(styles.PRED_INACTIVE)

        root_layout.addLayout(control_layout)
        root_layout.addWidget(styles.make_sep())
        sig_label = QLabel("Signals")
        sig_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        pred_label = QLabel("Predictions")
        pred_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        root_layout.addWidget(sig_label)
        root_layout.addLayout(sig_layout)
        root_layout.addWidget(styles.make_sep())
        root_layout.addWidget(pred_label)
        root_layout.addLayout(pred_layout)
    
    def closeEvent(self, event):
        if self.state.mr_connection is not None:
            self.state.mr_connection.stop.emit()
            self.state.mr_connection.wait()
        if self.state.ble_connection is not None:
            self.state.ble_connection.stop.emit()
            self.state.ble_connection.wait()
        if self.model_thread is not None:
            self.model_thread.stop.emit()
            self.model_thread.wait()
        super().closeEvent(event)
    
    def mindrove_connection_start(self):
        self.mindrove_connect.setEnabled(False)
        self.state.mr_connection = mr.MindRoveConnection()
        self.state.mr_connection.connected.connect(self.mindrove_connection_status)
        self.state.mr_connection.start()
    
    @Slot(int)
    def mindrove_connection_status(self, status):
        if status == mr.CONNECT_SUCCESS:
            self.mindrove_status_label_2.setText("connected")
            self.mindrove_status_label_2.setStyleSheet(styles.LABEL_YES)
            self.mindrove_connect.setText("Disconnect")
            self.mindrove_connect.disconnect(self.mindrove_connect_handle)
            self.mindrove_connect_handle = self.mindrove_connect.clicked.connect(self.mindrove_stop)
            self.mindrove_connect.setEnabled(True)
            for w in self.sigs:
                w.setMinimum(self.MIN_SIG)
                w.setMaximum(self.MAX_SIG)
                w.setEnabled(True)
            for w in self.preds:
                w.setStyleSheet(styles.PRED_DIM)
            self.state.mr_connection.update.connect(self.update_mr)
        else:
            self.mindrove_stop()

    def mindrove_stop(self):
        self.mindrove_connect.setEnabled(False)
        for i in range(8):
            self.sigs[i].setEnabled(False)
            self.sigs[i].setValue(0)
            self.sigs[i].setMinimum(0)
            self.sigs[i].setMaximum(1)
            self.sigs[i].setFormat(f"Channel {i + 1}")
        for w in self.preds:
            w.setStyleSheet(styles.PRED_INACTIVE)
        self.state.mr_connection.cleanup_complete.connect(self.mindrove_cleanup_complete)
        self.state.mr_connection.stop.emit()
    
    @Slot(int)
    def mindrove_cleanup_complete(self, status):
        self.mindrove_status_label_2.setText("not connected")
        self.mindrove_status_label_2.setStyleSheet(styles.LABEL_NO)
        self.mindrove_connect.disconnect(self.mindrove_connect_handle)
        self.mindrove_connect_handle = self.mindrove_connect.clicked.connect(self.mindrove_connection_start)
        self.mindrove_connect.setText("Connect")
        self.mindrove_connect.setEnabled(True)
        if status == mr.CONNECT_FAILURE:
            error = QMessageBox(self)
            error.setIcon(QMessageBox.Icon.Warning)
            error.setWindowTitle("Error")
            error.setText("MindRove connection failed.")
            error.setStandardButtons(QMessageBox.StandardButton.Ok)
            error.show()

    def arm_dialog_show(self):
        dialog = ArmDialog(self)
        dialog.complete.connect(self.arm_dialog_return)
        dialog.show()
    
    @Slot(object)
    def arm_dialog_return(self, address):
        self.arm_connect.setEnabled(False)
        self.state.ble_connection = ble.BLEConnection(address)
        self.state.ble_connection.connected.connect(self.arm_dialog_status)
        self.state.ble_connection.start()
    
    @Slot(int)
    def arm_dialog_status(self, status):
        if status == ble.CONNECT_SUCCESS:
            self.arm_status_label_2.setText("connected")
            self.arm_status_label_2.setStyleSheet(styles.LABEL_YES)
            self.arm_connect.setText("Disconnect")
            self.arm_connect.disconnect(self.arm_connect_handle)
            self.arm_connect_handle = self.arm_connect.clicked.connect(self.arm_dialog_stop)
            self.arm_connect.setEnabled(True)
        else:
            self.arm_dialog_stop()
    
    def arm_dialog_stop(self):
        self.arm_connect.setEnabled(False)
        self.state.ble_connection.cleanup_complete.connect(self.arm_dialog_cleanup_complete)
        self.state.ble_connection.stop.emit()
    
    @Slot(int)
    def arm_dialog_cleanup_complete(self, status):
        self.arm_status_label_2.setText("not connected")
        self.arm_status_label_2.setStyleSheet(styles.LABEL_NO)
        self.arm_connect.disconnect(self.arm_connect_handle)
        self.arm_connect_handle = self.arm_connect.clicked.connect(self.arm_dialog_show)
        self.arm_connect.setText("Connect")
        self.arm_connect.setEnabled(True)
        if status == ble.CONNECT_FAILURE:
            error = QMessageBox(self)
            error.setIcon(QMessageBox.Icon.Warning)
            error.setWindowTitle("Error")
            error.setText("Bluetooth connection failed.")
            error.setStandardButtons(QMessageBox.StandardButton.Ok)
            error.show()
    
    @Slot(object)
    def update_mr(self, data):
        last_row = data[-1, :]
        self.model_thread.deposit.emit(data)
        for i in range(8):
            self.sigs[i].setFormat(str(int(last_row[i])))
            self.sigs[i].setValue(max(min(self.MAX_SIG, last_row[i]), self.MIN_SIG))
        
    @Slot(int)
    def model_callback(self, i):
        if self.sigs[0].isEnabled():
            for w in self.preds:
                w.setStyleSheet(styles.PRED_DIM)
            if 0 <= i <= 5:
                self.preds[i].setStyleSheet(styles.PRED_LIT)
            if self.state.ble_connection is not None:
                self.state.ble_connection.deposit.emit(i)

if __name__ == "__main__":
    app = QApplication([])
    app.setApplicationDisplayName("Embrace")
    window = EmbraceApp()
    window.show()
    app.exec()