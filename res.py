import datetime
import os
import time
import threading
import psutil

ROOT_PATH = os.path.dirname(__file__)
RES_PATH = os.path.join(ROOT_PATH, "res")

if not os.path.exists(RES_PATH):
    os.mkdir(RES_PATH)

class ResThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.alive = True
        self.proc = psutil.Process(os.getpid())
        self.T = []
        self.M = []
        self.C = []
    
    def stop(self):
        self.alive = False
    
    def run(self):
        self.start_time = time.time()
        while self.alive:
            self.T.append(self.proc.num_threads())
            self.M.append(self.proc.memory_info().rss // 1024)
            self.C.append(time.time() - self.start_time)
            time.sleep(0.1)
        
        with open(os.path.join(RES_PATH, f"{datetime.datetime.now().isoformat(timespec="microseconds")}.txt"), "w") as f:
            f.write(" ".join(map(str, self.C)))
            f.write("\n")
            f.write(" ".join(map(str, self.T)))
            f.write("\n")
            f.write(" ".join(map(str, self.M)))
