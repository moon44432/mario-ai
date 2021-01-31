
import numpy as np
from collections import deque


class Memory():
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)