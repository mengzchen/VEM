import os
import sys
sys.path.append(os.getcwd())

from tqdm import tqdm
import random
from PIL import Image

import utils

system_prompt = """You are a helpful assistant. 
# Actions 
## You have the following actions. 
### Click 
Click: A quick, light fingertip press that commands, selects, or navigates through a phone's user interface. Parameters: {"point": "The specific point of interest on the screen, marked by the coordinate (x, y)."}
### Type 
Type: Engaging with a smartphone's interface by entering text for various purposes like messaging, searching, or command execution. Parameters: {"text": "The text to be typed on a smartphone."} 
### Select 
Select: click one point and select one choice. Parameters: {"point": "The specific point of interest on the screen, marked by the coordinate (x, y)", "choose": "the choice of selected item."} 
"""

def action2step(action, image_size):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER'], f"action: action_type"

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)

    # follow Qwen2VL setting
    click_point = [(point_x / image_size[0]) * 1000, (point_y / image_size[1]) * 1000]
    click_point = [int(item) for item in click_point]

    action_example = {}
    if action_type in ['CLICK', 'HOVER', 'ENTER']: 
        action_example["function"] = "click"
        action_example["args"] = {"point": click_point}
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_example["function"] = "select"
        action_example["args"] = {"point": click_point, "choose": select_value}
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_example["function"] = "select"
        action_example["args"] = {"point": click_point, "text": typed_text}

    return action_example


class Mind2Web:
    def __init__(self, dataset_name:str, split: str, date: str):
        self.dataset_name = dataset_name
        self.image_dir = f"data/images/{dataset_name}_images"
        self.split = split
        self.ann_rpath = f"data/{dataset_name}_anns/{dataset_name}_{split}.json"

        if not os.path.exists(f"data/{dataset_name}_anns/{date}"):
            os.mkdir(f"data/{dataset_name}_anns/{date}")
        self.ann_wpath = f"data/{dataset_name}_anns/{date}/{dataset_name}_{split}.json"
    
    def get_label_data(self):
        anns = utils.read_json(self.ann_rpath)

        utils.colorful_print(f"--- num of episode: {len(anns)}", "green")

        steps = []
        for episode in tqdm(anns):
            task = episode["confirmed_task"]
            ep_id = episode["annotation_id"]
            
            image_rpath_list, action_list = [], []
            conversations = [
                {"from": "user", "value": system_prompt + f"\nYour task is: {task}"}
            ]
            for step_id, step in enumerate(episode["actions"]):
                if "bbox" not in step:
                    utils.colorful_print("action not found", "red")
                    continue

                image_filename = ep_id + "-" + step["action_uid"] + ".jpg"
                image_rpath = os.path.join(self.image_dir, image_filename)
                if not os.path.exists(image_rpath):
                    utils.colorful_print(f"{image_rpath} image not found", "red")
                    continue
                image = Image.open(image_rpath)

                # add user screenshot
                conversations.append({
                    "from": "user",
                    "value": "screenshot: <image>"
                })

                # add ground truth
                action_step = action2step(step, image.size)
                conversations.append({
                    "from": "assistant",
                    "value": str(action_step)
                })
                
                image_rpath_list.append(image_rpath)

                steps.append({
                    "ep_id": ep_id, 
                    "step_id": step_id,
                    "task": task, 
                    "action_list": action_list,
                    "messages": conversations,
                    "images": image_rpath_list
                })

                action_list.append(action_step)
                
                # for m in steps[-1]["messages"]:
                #     print(m["content"])
                

        utils.colorful_print("--- Num of total step: " + str(len(steps)), "green")
        print(f"--- example\n{steps[:3]}")
        random.shuffle(steps)
        utils.write_json(steps, self.ann_wpath)

aitw_data = Mind2Web(dataset_name="mind2web", split="train", date="1023")
aitw_data.get_label_data()
