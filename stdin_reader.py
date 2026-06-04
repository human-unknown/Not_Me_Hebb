"""stdin_reader.py —— 非阻塞终端输入线程"""
import threading, queue

class StdinReader:
    def __init__(self):
        self.queue = queue.Queue()
        self._stop = False
        self._thread = None
    def start(self):
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    def _run(self):
        while not self._stop:
            try: self.queue.put(input())
            except (EOFError, OSError): break
    def stop(self): self._stop = True
