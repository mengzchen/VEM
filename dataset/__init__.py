import utils


class CogAgentDataset():
    def __init__(self, config, finish_task):
        origin_anns = utils.read_jsonl(config["data_path"])
        self.tasks = []
        for ann in origin_anns:
            if ann["task"] not in self.tasks and ann["task"] not in finish_task:
                self.tasks.append(ann["task"])
        print(f"\tlen of tasks: {len(self.tasks)}")
        self.query_format = "Task: {}\nHistory steps: {}\n(Platform: Mobile)\n(Answer in Action-Operation-Sensitive format.)\n"


    def __len__(self):
        return len(self.anns)


    def __getitem__(self, idx):
        return self.tasks[idx], self.query_format


def create_dataset(config, finish_task):
    if config["model_name"] == "cogagent":
        dataset = CogAgentDataset(config, finish_task)
        return dataset
    else:
        raise NotImplementedError(f"dataset == {dataset}")

    