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

        self.image_paths = None
        self.critic_inputs = None
        self.history_actions = None
        self.next_actions = None

    def sample(self, batch_size=None):
        rand_indices = np.random.randint(0, self.current_size, size=batch_size) % self.max_size
        return {
            "image_paths": self.image_paths[rand_indices],
            "critic_inputs": self.critic_inputs[rand_indices],
            "history_actions": self.history_actions[rand_indices],
            "next_actions": self.next_actions[rand_indices]
        }

    def __len__(self):
        return self.current_size

    def insert(
        self,
        image_path,
        critic_input,
        history_action,
        next_action,
        **kwargs
    ):
        if self.image_paths is None:
            self.image_paths = np.array([''] * self.max_size, dtype="object")
            self.critic_inputs = np.array([''] * self.max_size, dtype="object")
            self.history_actions = np.array([''] * self.max_size, dtype="object")
            self.next_actions = np.array([''] * self.max_size, dtype="object")

        self.image_paths[self.current_size % self.max_size] = image_path
        self.critic_inputs[self.current_size % self.max_size] = critic_input
        self.history_actions[self.current_size % self.max_size] = history_action
        self.next_actions[self.current_size % self.max_size] = next_action

        self.current_size += 1