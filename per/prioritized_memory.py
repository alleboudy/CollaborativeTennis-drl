import random
import numpy as np
from .SumTree import SumTree
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Memory:  # stored as ( s, a, r, s_,d ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.size = 0

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        self.size = min(self.size+1, self.capacity)
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        experiences = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            experiences.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        states = torch.from_numpy(
        np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e[1] for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
    def __len__(self):
        """Return the current size of internal memory."""
        return self.size