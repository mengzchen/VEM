from torch.utils.data import Dataset
import numpy as np


class CogAgentDataset():
    def __init__(self, anns):
        query_format = "Task: {}\nHistory steps: {}\n(Platform: Mobile)\n(Answer in Action-Operation-Sensitive format.)\n"
        self.anns = []
        for ann in anns:
            task = ann["task"]
            
            history_list = []
            for i in range(ann["step_id"]):
                action_type = ann["action_list"][i]["action_type"]
                touch_point = ann["action_list"][i]["touch_point"]
                x, y = int(touch_point[0] * 1000), int(touch_point[1] * 1000)
                if action_type == "DUAL_POINT":
                    desc = f"CLICK(box=[[{x},{y},{x},{y}]])"
                elif action_type == "TYPE":
                    desc = f"TYPE(box=[[{x},{y},{x},{y}]], text='{ann['action_list'][i]['typed_text']}')"
                elif "SCROLL" in action_type:
                    desc = f"{action_type}(box=[[{x},{y},{x},{y}]], step_count=5)"
                elif action_type == "PRESS_ENTER":
                    desc = f"KEY_PRESS(key='Enter')"
                elif action_type == "PRESS_HOME":
                    desc = f"KEY_PRESS(key='Home')"
                elif action_type == "PRESS_BACK":
                    desc = f"KEY_PRESS(key='Back')"
                elif action_type == "STATUS_TASK_COMPLETE":
                    desc = "END()"
                else:
                    print(action_type)

                history_list.append(f"{i}.{desc}")
            self.anns.append({
                "ep_id": ann["ep_id"],
                "step_id": ann["step_id"],
                "policy_input": query_format.format(task, "\n".join(history_list)),
                "policy_image": ann["policy_image"],
                "policy_output": ann["policy_output"]
            })


    def __len__(self):
        return len(self.anns)


    def __getitem__(self, idx):
        return self.anns[idx]
    

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
        self.action_lists = None
        self.tasks = None
        self.step_ids = None

    def sample(self, batch_size=None):
        rand_indices = np.random.randint(0, self.current_size, size=batch_size) % self.max_size
        return {
            "critic_images": self.critic_images[rand_indices],
            "critic_inputs": self.critic_inputs[rand_indices],
            "policy_outputs": self.policy_outputs[rand_indices],
            "policy_inputs": self.policy_inputs[rand_indices],
            "policy_images": self.policy_images[rand_indices],
            "action_lists": self.action_lists[rand_indices],
            "tasks": self.tasks[rand_indices],
            "step_ids": self.step_ids[rand_indices]
        }

    def __len__(self):
        return self.current_size

    def insert(
        self,
        policy_output,
        policy_input,
        policy_image,
        action_list,
        task,
        step_id, 
        critic_image="",
        critic_input="",
        **kwargs
    ):
        if self.critic_images is None:
            self.critic_images = np.array([''] * self.max_size, dtype="object")
            self.critic_inputs = np.array([''] * self.max_size, dtype="object")
            self.policy_outputs = np.array([''] * self.max_size, dtype="object")
            self.policy_inputs = np.array([''] * self.max_size, dtype="object")
            self.policy_images = np.array([''] * self.max_size, dtype="object")
            self.action_lists = np.array([''] * self.max_size, dtype="object")
            self.tasks = np.array([''] * self.max_size, dtype="object")
            self.step_ids = np.array([''] * self.max_size, dtype="object")

        self.critic_images[self.current_size % self.max_size] = critic_image
        self.critic_inputs[self.current_size % self.max_size] = critic_input
        self.policy_outputs[self.current_size % self.max_size] = policy_output
        self.policy_inputs[self.current_size % self.max_size] = policy_input
        self.policy_images[self.current_size % self.max_size] = policy_image
        self.action_lists[self.current_size % self.max_size] = action_list
        self.tasks[self.current_size % self.max_size] = task
        self.step_ids[self.current_size % self.max_size] = step_id

        self.current_size += 1


def qwen_action2step(step_data):
    action_type = step_data["action_type"]
    
    if action_type == "DUAL_POINT":
        touch_point, lift_point = step_data["touch_point"], step_data["lift_point"]
        click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]
        click_point = [f"{item:.2f}" for item in click_point]
        click_point = "({},{})".format(click_point[0], click_point[1])
        action = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
    elif action_type == 'SCROLL_DOWN':
        action_type_new = 0
        action = "{{\"action_type\": {}}}".format(action_type_new)
    elif action_type == 'SCROLL_UP':
        action_type_new = 1
        action = "{{\"action_type\": {}}}".format(action_type_new)
    elif action_type == 'SCROLL_LEFT':
        action_type_new = 8
        action = "{{\"action_type\": {}}}".format(action_type_new)
    elif action_type == 'SCROLL_RIGHT':
        action_type_new = 9
        action = "{{\"action_type\": {}}}".format(action_type_new)
    elif action_type == "TYPE":
        action = "{{\"action_type\": {}, \"typed_text\": \"{}\"}}".format(3, step_data["typed_text"])
    elif action_type == "PRESS_HOME":
        action = "{{\"action_type\": {}}}".format(6)
    elif action_type == "PRESS_BACK":
        action = "{{\"action_type\": {}}}".format(5)
    elif action_type == "PRESS_ENTER":
        action = "{{\"action_type\": {}}}".format(7)
    elif action_type == "STATUS_TASK_COMPLETE":
        action = "{{\"action_type\": {}}}".format(10)
    elif action_type == "STATUS_TASK_IMPOSSIBLE":
        action = "{{\"action_type\": {}}}".format(11)
    else:
        action = "{{\"action_type\": {}}}".format(6)

    return action


class Qwen2VLDataset():
    def __init__(self, anns, is_train):
        if is_train is False:
            query_format = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
            self.anns = []
            for ann in anns:
                task, history = ann["task"], []
                for i, step in enumerate(ann["action_list"]):
                    format_action = qwen_action2step(step)
                    history.append('Step' + str(i) + ': ' + format_action + ". ") 
                    
                self.anns.append({
                    "ep_id": ann["ep_id"],
                    "step_id": ann["step_id"],
                    "policy_input": query_format.format(task, "".join(history[:ann["step_id"]])),
                    "policy_image": ann["policy_image"],
                    "policy_output": ann["policy_output"]
                })
        else:
            query_format = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
            self.anns = []
            for ann in anns:
                task, history = ann["tasks"], []
                for i, step in enumerate(ann["action_lists"]):
                    format_action = qwen_action2step(step)
                    history.append('Step' + str(i) + ': ' + format_action + ". ") 
                
                self.anns.append({
                    "policy_inputs": query_format.format(task, "".join(history[:ann["step_ids"]])),
                    "policy_images": ann["policy_images"],
                    "policy_outputs": ann["policy_outputs"],
                    "critic_inputs": ann["critic_inputs"],
                    "critic_images": ann["critic_images"]
                })


    def __len__(self):
        return len(self.anns)


    def __getitem__(self, idx):
        return self.anns[idx]