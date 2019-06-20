from threading import Thread
import time
import GPUtil
import psutil

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.recorded_gpu = []
        self.recorded_cpu = []
        self.recorded_memory = []
        self.start()

    def run(self):
        gpu = GPUtil.getGPUs().pop()
        while not self.stopped:
            self.recorded_gpu.append(gpu.load)
            self.recorded_cpu.append(psutil.cpu_percent())
            self.recorded_memory.append(psutil.virtual_memory().percent)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True