from torch.utils.data import Dataset, DataLoader
import utils


class CogAgentDataset(Dataset):
    def __init__(self, config):
        self.anns = utils.read_jsonl(config["data_path"])
        self.query_format = "Task: {}\nHistory steps: {}\n(Platform: Mobile)\n(Answer in Action-Operation-Sensitive format.)\n"


    def __len__(self):
        return len(self.anns)


    def __getitem__(self, idx):
        ann = self.anns[idx]
        # TODO no history
        text = self.query_format.format(ann["task"], "")
        
        return text


def create_dataset(config):
    if config["model_name"] == "cogagent":
        dataset = CogAgentDataset(config)
        return dataset
    else:
        raise NotImplementedError(f"dataset == {dataset}")

    