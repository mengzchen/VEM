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
        history = ""
        for index, (grounded_op_func, action) in enumerate(zip(history_grounded_op_funcs, history_actions)):
            history += f"\n{index}. {grounded_op_func}\t{action}"
        text = self.query_format.format(ann["task"], history)
        
        return text, image


def create_dataset(config):
    if config["model_name"] == "cogagent":
        dataset = CogAgentDataset(config)
        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
        )

        return dataloader
    else:
        raise NotImplementedError(f"dataset == {dataset}")

    