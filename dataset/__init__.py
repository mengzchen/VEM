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


class AutoGUIDataset():
    def __init__(self, config, finish_task):
        self.query_format = "Previous Action:\n{}\nGoal:\n{}"
        origin_anns = utils.read_jsonl(config["data_path"])

        self.anns, tasks = [], []
        for ann in origin_anns:
            if ann["task"] not in tasks and ann["task"] not in finish_task.keys():
                self.anns.append({"task": ann["task"], "task_id": ann["ep_id"]})
                tasks.append(ann["task"])

        print(f"\tlen of tasks: {len(self.anns)}")
        

    def __len__(self):
        return len(self.anns)


    def __getitem__(self, idx):
        return self.anns[idx]["task_id"], self.anns[idx]["task"], self.query_format


def create_dataset(config, finish_task):
    if config["model_name"] == "cogagent":
        dataset = CogAgentDataset(config, finish_task)
        return dataset
    elif config["model_name"] == "autogui":
        dataset = AutoGUIDataset(config, finish_task)
        return dataset
    else:
        raise NotImplementedError(f"dataset == {dataset}")

    