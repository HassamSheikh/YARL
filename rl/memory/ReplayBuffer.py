from collections import deque
import random
class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer=deque(maxlen=size)

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)

    def add_to_buffer(self, experience):
        self.buffer.append(experience)
