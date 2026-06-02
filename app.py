from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QMovie, QKeySequence
from threading import Lock
from collections import deque
import ble
import mr
import models
import tele
import styles
import os
import datetime
import numpy as np
import time

ROOT_PATH = os.path.dirname(__file__)
RECORD_PATH = os.path.join(ROOT_PATH, "record")
ASSET_PATH = os.path.join(ROOT_PATH, "assets")
if not os.path.exists(RECORD_PATH):
    os.mkdir(RECORD_PATH)
MAX_SIG = 1e5
MIN_SIG = -1e5
GESTURES = ["Extend", "Fist", "Flex", "Pronation", "Radial", "Rest", "Supination", "Ulnar"]

DEBUG_RES = os.getenv("EMBRACE_RES") == "1"

GLOBAL_GARBAGE = []

def time_ms():
    return int(time.time() * 1000)

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
    
    def closeEvent(self, event):
        try:
            self._thread.stop.emit()
            self._thread.wait()
        except:
            pass
        super().closeEvent(event)

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
        self.curr_movie = QLabel()
        self.curr_movie.setFixedHeight(400)
        self.curr_movie.setFixedWidth(300)
        self.curr_movie.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.curr_movie_obj = None
        curr_movie_layout = QHBoxLayout()
        curr_movie_layout.addWidget(self.curr_movie, Qt.AlignCenter)
        self.next = QLabel("--")
        self.next.setStyleSheet("font-size: 15px;")
        self.next.setAlignment(Qt.AlignCenter)
        self.counter = QLabel("--")
        self.counter.setStyleSheet("font-size: 15px;")
        self.counter.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(self.curr)
        root_layout.addLayout(curr_movie_layout)
        root_layout.addWidget(self.next)
        root_layout.addWidget(self.counter)
        self.control_lim = QCheckBox("Stop after 5 rounds")
        self.control = QPushButton("Start")
        self.control.clicked.connect(self.record_control)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        self.ft_button = QPushButton("Fine Tune")
        self.ft_button.clicked.connect(self.fine_tune)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        root_layout.addWidget(self.control_lim)
        root_layout.addWidget(self.control)
        root_layout.addWidget(self.save_button)
        root_layout.addWidget(self.ft_button)
        root_layout.addWidget(self.close_button)

        self.recording = False
        self.lock = Lock()
        self.deposit.connect(self.handle_deposit)
        self.app.model_thread.fine_tune_end.connect(self.fine_tune_end)

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

    def fine_tune(self):
        self.control.setEnabled(False)
        self.save_button.setEnabled(False)
        self.ft_button.setEnabled(False)
        if len(np.unique(self.memory[:, -1])) == len(self.app.state.gestures) + 1:
            self.app.model_thread.fine_tune.emit(self.memory)
        else:
            error = QMessageBox(self)
            error.setIcon(QMessageBox.Icon.Warning)
            error.setWindowTitle("Warning")
            error.setText("Fine tuning requires you to record every gesture at least once.")
            error.setStandardButtons(QMessageBox.StandardButton.Ok)
            error.show()
            self.control.setEnabled(True)
            self.save_button.setEnabled(True)
            self.ft_button.setEnabled(True)
    
    def fine_tune_end(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Success")
        msg.setText("Fine tuning has finished.")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.show()
        self.control.setEnabled(True)
        self.save_button.setEnabled(True)
        self.ft_button.setEnabled(True)
        self.memory = np.empty((0, 8))
        self.counter.setText("Samples: 0")

    @Slot(object)
    def handle_deposit(self, data):
        with self.lock:
            if self.active_recording:
                self.memory = np.vstack((self.memory, data))
                self.labels = np.concatenate((self.labels, np.repeat(self.curr_index if self.curr_index >= 0 else np.nan, len(data))))
    
    def record_control(self):
        if not self.recording:
            with self.lock:
                self.memory = np.empty((0, 8))
                self.labels = np.array([])
            self.app.record_thread = mr.MindRoveRecord(len(self.app.state.gestures), self.control_lim.checkState() == Qt.Checked)
            self.app.record_thread.instruction.connect(self.instruction_callback)
            self.app.record_thread.end.connect(self.record_stop_wait)
            self.save_button.setEnabled(False)
            self.ft_button.setEnabled(False)
            self.close_button.setEnabled(False)
            self.control_lim.setEnabled(False)
            self.control.setText("Stop")
            self.counter.setText("--")
            self.recording = True
            self.app.record_thread.start()
        else:
            self.app.record_thread.stop.emit()
    
    def record_stop_wait(self):
        with self.lock:
            self.active_recording = False
            self.curr_index = -1
            self.memory = np.hstack((self.memory, self.labels.reshape(-1, 1)))

        self.recording = False
        self.curr.setText("--")
        self.next.setText("--")
        self.control.setText("Start")
        self.control_lim.setEnabled(True)
        self.save_button.setEnabled(True)
        self.ft_button.setEnabled(True)
        self.close_button.setEnabled(True)
    
    def _get_gesture_name(self, i):
        if i < 0:
            return "PAUSE"
        return self.app.state.gestures[i]

    @Slot(object)
    def instruction_callback(self, instruction):
        if instruction[0] == "cd":
            self.curr.setText(str(instruction[1]))
            self.next.setText(f"Next: {self._get_gesture_name(instruction[2])}")

            if instruction[1] == 3:
                self.curr_movie_obj = QMovie(os.path.join(ASSET_PATH, f"{self._get_gesture_name(instruction[2])}.gif"))
                self.curr_movie.setMovie(self.curr_movie_obj)
                self.curr_movie_obj.start()
                self.curr_movie.show()
        elif instruction[0] == "use":
            with self.lock:
                self.active_recording = True
                self.curr_index = instruction[1]

            self.curr.setText(f"{self._get_gesture_name(instruction[1])} {instruction[3]}")
            if instruction[1] < 0 and instruction[3] == 3:
                self.curr_movie_obj = QMovie(os.path.join(ASSET_PATH, f"{self._get_gesture_name(instruction[2])}.gif"), parent=self)
                self.curr_movie.setMovie(self.curr_movie_obj)
                self.curr_movie_obj.start()
                self.curr_movie.show()
            elif instruction[1] >= 0 and instruction[3] == 1:
                try:
                    self.curr_movie.stop()
                except:
                    pass
            self.next.setText(f"Next: {self._get_gesture_name(instruction[2])}")
            with self.lock:
                self.counter.setText(f"Samples: {len(self.memory)}")

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
        self.model_choose_menu.currentTextChanged.connect(self.change_model)
        self.model_thread.set_model(self.model_choose_menu.currentText())
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
        self.pred_sims = [QPushButton("Test") for _ in self.state.gestures]
        for i in range(min(len(self.pred_sims), 9)):
            self.pred_sims[i].setShortcut(QKeySequence(str(i + 1)))
        pred_layout = QHBoxLayout()
        for l, t in zip(self.preds, self.pred_sims):
            pred_layout_each = QVBoxLayout()
            pred_layout_each.addWidget(l)
            l.setFixedWidth(100)
            l.setFixedHeight(100)
            l.setAlignment(Qt.AlignCenter)
            l.setStyleSheet(styles.PRED_INACTIVE)
            pred_layout_each.addWidget(t)
            t.setEnabled(False)
            pred_layout.addLayout(pred_layout_each)

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

        root_layout.addWidget(styles.make_sep())
        opt_label = QLabel("Options")
        opt_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        root_layout.addWidget(opt_label)
        opt_layout = QHBoxLayout()
        self.opt_dedup = QCheckBox("Send duplicate gestures")
        opt_layout.addWidget(self.opt_dedup)
        root_layout.addLayout(opt_layout)

        self.record_blocking = False

        self.arm_test_callables = [self._arm_test_callable(i) for i in range(len(self.state.gestures))]

        self.recent_predictions = deque()
        self.last_predicted = -1

        self.tele_thread = tele.TeleThread(self)
        self.tele_thread.start()
    
    def _arm_test_callable(self, i):
        def _arm_test_callable_inner():
            if self.state.ble_connection is not None:
                self.state.ble_connection.deposit.emit(i)
        return _arm_test_callable_inner
    
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
        if self.tele_thread is not None:
            self.tele_thread.stop.emit()
            self.tele_thread.wait()
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
        GLOBAL_GARBAGE.append(self.state.mr_connection)
        self.state.mr_connection = None
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
            for i in range(len(self.pred_sims)):
                self.pred_sims[i].clicked.connect(self.arm_test_callables[i])
                self.pred_sims[i].setEnabled(True)
    
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
        GLOBAL_GARBAGE.append(self.state.ble_connection)
        self.state.ble_connection = None
        self.last_predicted = -1
        for t in self.pred_sims:
            t.setEnabled(False)
        if status == ble.CONNECT_FAILURE:
            error = QMessageBox(self)
            error.setIcon(QMessageBox.Icon.Warning)
            error.setWindowTitle("Error")
            error.setText("Bluetooth connection failed.")
            error.setStandardButtons(QMessageBox.StandardButton.Ok)
            error.show()
    
    @Slot(object)
    def update_mr(self, data):
        last_row = data[-1, :] # only one sample displayed on GUI
        if self.record_blocking:
            for w in self.preds:
                w.setStyleSheet(styles.PRED_INACTIVE)
            self.record_dialog.deposit.emit(data)
        else:
            self.model_thread.deposit.emit(data)
        for i in range(8):
            self.sigs[i].setFormat(str(int(last_row[i])))
            self.sigs[i].setValue(max(min(MAX_SIG, last_row[i]), MIN_SIG))
        
    @Slot(object)
    def model_callback(self, data):
        i, v = data
        if not self.record_blocking:
            for w in self.preds:
                w.setStyleSheet(styles.PRED_DIM)
            if 0 <= i < len(self.state.gestures) and v > 0.9:
                self.recent_predictions.append(i)
                while len(self.recent_predictions) > 20:
                    self.recent_predictions.popleft()
                votes = 0
                for j in self.recent_predictions:
                    if j == i:
                        votes += 1
                if (self.last_predicted == i and votes >= 12) or (self.last_predicted != i and votes >= 18):
                    if self.state.ble_connection is not None and (i != self.last_predicted or self.opt_dedup.checkState() == Qt.Checked):
                        self.last_predicted = i
                        self.state.ble_connection.deposit.emit(i)
                    self.preds[i].setStyleSheet(styles.PRED_HIGH)
    
    def mindrove_record_start(self):
        self.record_blocking = True
        self.record_dialog = RecordDialog(self)
        self.record_dialog.finished.connect(self.mindrove_record_callback)
        self.record_dialog.show()

    def mindrove_record_callback(self):
        self.record_blocking = False

if __name__ == "__main__":
    if DEBUG_RES:
        import res
        res_thread = res.ResThread()
        res_thread.start()
    app = QApplication([])
    app.setApplicationDisplayName("Embrace")
    window = EmbraceApp()
    window.show()
    app.exec()
    if DEBUG_RES:
        res_thread.stop()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(res_thread.C, res_thread.T, color="tab:red", label="Thread Count")
        ax.set_xlabel("Time")
        ax.set_ylabel("Thread Count")
        axx = ax.twinx()
        axx.plot(res_thread.C, res_thread.M, color="tab:blue", label="Memory (KB)")
        axx.set_ylabel("Memory (KB)")
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
        plt.suptitle("Embrace Resources Monitor")
        plt.show()