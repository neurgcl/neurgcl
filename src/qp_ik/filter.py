# a moving average 3d vel filter class
from copy import deepcopy
import threading

import numpy as np


class AverageFilter:
    def __init__(self, size=10, n_dim=3):
        self.lock = threading.Lock()
        self.size = size
        self.buffer = np.zeros((size, n_dim))
        self.idx = 0
        self.full = False

    def reset(self):
        with self.lock:
            self.buffer = np.zeros_like(self.buffer)
            self.idx = 0
            self.full = False

    def update(self, val):
        with self.lock:
            self.buffer[self.idx] = val
            self.idx += 1
            if self.idx >= self.size:
                self.idx = 0
                self.full = True

    def get_buffer(self):
        with self.lock:
            return deepcopy(self.buffer)

    def get(self):
        return self.get_mean()

    def get_mean(self):
        with self.lock:
            if self.full:
                return np.mean(self.buffer, axis=0)
            else:
                return np.mean(self.buffer[: self.idx], axis=0)

    def get_std(self):
        with self.lock:
            if self.full:
                return np.std(self.buffer, axis=0)
            else:
                return np.std(self.buffer[: self.idx], axis=0)

    def get_maxabs(self):
        with self.lock:
            if self.full:
                return np.max(np.abs(self.buffer), axis=0)
            else:
                return np.max(np.abs(self.buffer[: self.idx]), axis=0)
