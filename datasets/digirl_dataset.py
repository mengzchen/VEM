from torch.utils.data import Dataset
import numpy as np


class DummyDataset(Dataset):
    def __init__(self, anns, keys):
        self.anns = []
        for ann in anns:
            self.anns.append({k:v for k, v in ann.items() if k in keys})

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        return self.anns[idx]


class ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.max_size = capacity
        self.current_size = 0   
        self.batch_size = batch_size

        self.critic_images = None
        self.critic_inputs = None
        self.policy_outputs = None
        self.policy_inputs = None
        self.policy_images = None

    def sample(self, batch_size=None):
        rand_indices = np.random.randint(0, self.current_size, size=batch_size) % self.max_size
        return {
            "critic_images": self.critic_images[rand_indices],
            "critic_inputs": self.critic_inputs[rand_indices],
            "policy_outputs": self.policy_outputs[rand_indices],
            "policy_inputs": self.policy_inputs[rand_indices],
            "policy_images": self.policy_images[rand_indices]
        }

    def __len__(self):
        return self.current_size

    def insert(
        self,
        critic_image,
        critic_input,
        policy_output,
        policy_input,
        policy_image,
        **kwargs
    ):
        if self.critic_images is None:
            self.critic_images = np.array([''] * self.max_size, dtype="object")
            self.critic_inputs = np.array([''] * self.max_size, dtype="object")
            self.policy_outputs = np.array([''] * self.max_size, dtype="object")
            self.policy_inputs = np.array([''] * self.max_size, dtype="object")
            self.policy_images = np.array([''] * self.max_size, dtype="object")

        self.critic_images[self.current_size % self.max_size] = critic_image
        self.critic_inputs[self.current_size % self.max_size] = critic_input
        self.policy_outputs[self.current_size % self.max_size] = policy_output
        self.policy_inputs[self.current_size % self.max_size] = policy_input
        self.policy_images[self.current_size % self.max_size] = policy_image

        self.current_size += 1