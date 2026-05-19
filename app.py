from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Signal, Slot
from threading import Lock
import ble
import mr
import models
import styles
import os
import datetime
import numpy as np

ROOT_PATH = os.path.dirname(__file__)
RECORD_PATH = os.path.join(ROOT_PATH, "record")
if not os.path.exists(RECORD_PATH):
    os.mkdir(RECORD_PATH)
MAX_SIG = 1e5
MIN_SIG = -1e5
#GESTURES = ["Extend", "Fist", "Flex", "Radial", "Rest", "Ulnar"]
GESTURES = ["Extend", "Fist", "Flex", "Pronation", "Radial", "Rest", "Supination", "Ulnar"]

class EmbraceState:
    def __init__(self, gestures):
        self.mindrove = None
        self.arm = None
        self.model_manager = models.ModelManager()
        self.ble_connection = None
        self.mr_connection = None
        self.gestures = gestures

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

class RecordDialog(QDialog):
    deposit = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowModality(Qt.WindowModal)
        self.setWindowTitle("Record MindRove")
        
        self.setMinimumWidth(400)
        self.app = parent
        root_layout = QVBoxLayout(self)
        self.curr = QLabel("--")
        self.curr.setStyleSheet("font-size: 30px;")
        self.curr.setAlignment(Qt.AlignCenter)
        self.next = QLabel("--")
        self.next.setStyleSheet("font-size: 15px;")
        self.next.setAlignment(Qt.AlignCenter)
        self.counter = QLabel("--")
        self.counter.setStyleSheet("font-size: 15px;")
        self.counter.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(self.curr)
        root_layout.addWidget(self.next)
        root_layout.addWidget(self.counter)
        self.control = QPushButton("Start")
        self.control.clicked.connect(self.record_control)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        root_layout.addWidget(self.control)
        root_layout.addWidget(self.save_button)
        root_layout.addWidget(self.close_button)

        self.recording = False
        self.lock = Lock()
        self.deposit.connect(self.handle_deposit)

        self.memory = np.empty((0, 8))
        self.labels = np.array([])
        self.active_recording = False
        self.curr_index = -1
    
    def save(self):
        if len(self.memory) > 0:
            fn = f"{datetime.datetime.now().isoformat(timespec="microseconds")}.csv"
            fn = fn.replace(":", "-")
            np.savetxt(os.path.join(RECORD_PATH, fn) , self.memory, delimiter="\t")
            self.memory = np.empty((0, 8))
            self.counter.setText("Samples: 0")
        else:
            error = QMessageBox(self)
            error.setIcon(QMessageBox.Icon.Warning)
            error.setWindowTitle("Warning")
            error.setText("No samples remaining to save.")
            error.setStandardButtons(QMessageBox.StandardButton.Ok)
            error.show()

    @Slot(object)
    def handle_deposit(self, data):
        self.lock.acquire()
        if self.active_recording:
            self.memory = np.vstack((self.memory, data))
            self.labels = np.concatenate((self.labels, np.repeat(self.curr_index, len(data))))
        self.lock.release()
    
    def record_control(self):
        if not self.recording:
            self.lock.acquire()
            self.memory = np.empty((0, 8))
            self.labels = np.array([])
            self.lock.release()
            self.app.record_thread = mr.MindRoveRecord(len(self.app.state.gestures))
            self.app.record_thread.instruction.connect(self.instruction_callback)
            self.app.record_thread.end.connect(self.record_stop_wait)
            self.save_button.setEnabled(False)
            self.close_button.setEnabled(False)
            self.control.setText("Stop")
            self.counter.setText("--")
            self.recording = True
            self.app.record_thread.start()
        else:
            self.app.record_thread.stop.emit()
    
    def record_stop_wait(self):
        self.lock.acquire()
        self.active_recording = False
        self.curr_index = -1
        self.memory = np.hstack((self.memory, self.labels.reshape(-1, 1)))
        self.lock.release()

        self.recording = False
        self.curr.setText("--")
        self.next.setText("--")
        self.control.setText("Start")
        self.save_button.setEnabled(True)
        self.close_button.setEnabled(True)

    @Slot(object)
    def instruction_callback(self, instruction):
        if instruction[0] == "cd":
            self.curr.setText(str(instruction[1]))
            self.next.setText(f"Next: {self.app.state.gestures[instruction[2]]}")
        elif instruction[0] == "use":
            self.lock.acquire()
            self.active_recording = True
            self.curr_index = instruction[1]
            self.lock.release()

            self.curr.setText(f"{self.app.state.gestures[instruction[1]]} {instruction[3]}")
            self.next.setText(f"Next: {self.app.state.gestures[instruction[2]]}")
            self.lock.acquire()
            self.counter.setText(f"Samples: {len(self.memory)}")
            self.lock.release()

    def closeEvent(self, event):
        if self.app.record_thread is not None:
            self.app.record_thread.stop.emit()
            self.app.record_thread.wait()
        super().closeEvent(event)

class EmbraceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.state = EmbraceState(GESTURES)
        self.model_thread = models.ModelThread(self.state.model_manager)
        self.model_thread.predicted.connect(self.model_callback)

        self.record_thread = None

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
        mindrove_ops = QVBoxLayout()
        self.mindrove_connect = QPushButton("Connect")
        self.mindrove_connect_handle = self.mindrove_connect.clicked.connect(self.mindrove_connection_start)
        self.mindrove_record = QPushButton("Record")
        self.mindrove_record.clicked.connect(self.mindrove_record_start)
        self.mindrove_record.setEnabled(False)
        mindrove_ops.addWidget(self.mindrove_connect)
        mindrove_ops.addWidget(self.mindrove_record)
        control_layout.addLayout(mindrove_ops)
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
        self.model_choose_menu = QComboBox()
        self.model_choose_menu.addItems(models.MODEL_CONFIG.keys())
        #self.model_choose_menu.currentIndexChanged(self.change_model)
        self.state.model_manager.set_model(self.model_choose_menu.currentText())
        self.model_thread.start()
        model_choose.addWidget(model_choose_label)
        model_choose.addWidget(self.model_choose_menu)
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

        self.preds = [QLabel(g) for g in self.state.gestures]
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

        self.record_blocking = False
    
    def closeEvent(self, event):
        if self.record_thread is not None:
            self.record_thread.stop.emit()
            self.record_thread.wait()
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
    
    @Slot(int)
    def change_model(self, index):
        self.state.model_manager.set_model(self.model_choose_menu.currentText())
    
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
                w.setMinimum(MIN_SIG)
                w.setMaximum(MAX_SIG)
                w.setEnabled(True)
            for w in self.preds:
                w.setStyleSheet(styles.PRED_DIM)
            self.mindrove_record.setEnabled(True)
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
        self.mindrove_record.setEnabled(False)
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
        if self.record_blocking:
            self.record_dialog.deposit.emit(data)
        else:
            self.model_thread.deposit.emit(data)
        for i in range(8):
            self.sigs[i].setFormat(str(int(last_row[i])))
            self.sigs[i].setValue(max(min(MAX_SIG, last_row[i]), MIN_SIG))
        
    @Slot(int)
    def model_callback(self, i):
        if self.mindrove_record.isEnabled():
            for w in self.preds:
                w.setStyleSheet(styles.PRED_DIM)
            if 0 <= i < len(self.state.gestures):
                self.preds[i].setStyleSheet(styles.PRED_LIT)
            if self.state.ble_connection is not None:
                self.state.ble_connection.deposit.emit(i)
    
    def mindrove_record_start(self):
        self.record_blocking = True
        self.record_dialog = RecordDialog(self)
        self.record_dialog.finished.connect(self.mindrove_record_callback)
        self.record_dialog.show()

    def mindrove_record_callback(self):
        self.record_blocking = False

if __name__ == "__main__":
    app = QApplication([])
    app.setApplicationDisplayName("Embrace")
    window = EmbraceApp()
    window.show()
    app.exec()