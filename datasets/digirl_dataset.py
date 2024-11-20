from torch.utils.data import Dataset
import numpy as np


class DummyDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


class ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.max_size = capacity
        self.current_size = 0   
        self.batch_size = batch_size

        """
        observations
            Previous Actions: []
            Goal: str
        actions
            Action Plan: []
            Action Decision: str
        """
        self.observations = None
        self.actions = None
        self.image_features = None

    def sample(self, batch_size=None):
        rand_indices = np.random.randint(0, self.current_size, size=batch_size) % self.max_size
        return {
            "observation": self.observations[rand_indices],
            "action": self.actions[rand_indices],
            "image_features": self.image_features[rand_indices]
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        observation, 
        action,
        image_features
    ):
        if self.observations is None:
            self.observations = np.array([''] * self.max_size, dtype='object')
            self.actions = np.array([''] * self.max_size, dtype='object')
            self.image_features = np.empty((self.max_size, *image_features.shape), dtype=image_features.dtype)

        self.observations[self.size % self.max_size] = observation 
        self.image_features[self.size % self.max_size] = image_features
        self.actions[self.size % self.max_size] = action

        self.size += 1