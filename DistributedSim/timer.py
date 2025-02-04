import time
from contextlib import ContextDecorator

class Timer(ContextDecorator):
    def __enter__(self):
        self.start = time.perf_counter()
        return self  # Allows `as` keyword to access timing attributes

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        print(f"Elapsed time: {self.elapsed:.6f} seconds")
