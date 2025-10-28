import time


class Timer:
    """代码运行计时"""

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"Elapsed time: {self.elapsed_time:.6f} seconds")
