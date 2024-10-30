import os
import sys
sys.path.append(os.getcwd())

from tqdm import tqdm
from typing import List

import utils

system_prompt = """Task: Evaluate the prediction of Current action based on the given GUI screenshot and task requirements.

Instructions:
1. Review the history Action: Carefully examine the provided sequence of actions.
2. Assess the Task Requirements: Understand what the task is asking for.
3. Evaluate the Prediction of Current Action: Based on the provided entire actions and GUI screenshot, determine if current action aligns with the task requirements.
4. Rate the Action: Provide a rating based on four levels:
   - Level 1: The action is significantly deviating from the task requirements.
   - Level 2: The action might lead to a non-ideal page but is somewhat aligned with the task direction.
   - Level 3: The action contributes to the task requirements but may not be the most efficient path.
   - Level 4: The action is the optimal step towards achieving the task requirements.     

Explanation of Action:
- `action_type`: includes 'click', 'scroll down', 'scroll up', 'status task complete' (indicates the task is completed), 'press home' (return to the home page), 'press back', 'type' (typing text)
- `click_point`: if the action_type is click, this key will provide the relative position on the screenshot, (x, y) between [0, 1]
- `typed_text`: if the action_type is type, this key will provide the content.

Output Format:
[1-4]

Example Input:
Task Requirements: User needs to navigate to the settings page and enable Dark Mode. 
History Action:
Step 0: action_type is press home.
Current Action: 
Step 1: action_type is click, click_point is (0.3, 0.8).

Example Output:
2

Task Requirements: {}
History Action: 
{}
Current Action: 
{}
"""

def action2step(step, image_rpath):
    action_type = step["action_type_id"]
    action_type_text = step["action_type_text"]

    if action_type == 4:
        if action_type_text == "click":  
            touch_point = step["touch"]
            lift_point = step["lift"]
            click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]

            # add point in screenshot
            image_rpath = utils.add_visilize2screenshot(image_rpath=image_rpath, action_type="click", action_params=click_point)

            click_point = [f"{item:.2f}" for item in click_point]
            click_point = "({},{})".format(click_point[0], click_point[1])
            action = "action_type is {}, click_point is {}.".format(action_type_text, click_point)
        else: 
            action = "action_type is {}.".format(action_type_text)
    elif action_type == 3:
        action = "action_type is {}, typed_text is {}.".format(action_type_text, step["type_text"])
    else:
        action = "action_type is {}.".format(action_type_text)

    return action, image_rpath


class AITW:
    def __init__(self, split: str, parts: List, date: str):
        self.image_dir = "data/images/aitw_images"
        self.split = split
        self.ann_rpath = f"data/aitw_anns/aitw_{split}.json"

        # choice: general single webshopping install googleapps
        self.parts = parts
        if not os.path.exists(f"data/aitw_anns/{date}"):
            os.mkdir(f"data/aitw_anns/{date}")
        self.ann_wpath = f"data/aitw_anns/{date}/aitw_{split}.json"
    
    def get_label_data(self):
        all_anns = utils.read_json(self.ann_rpath)

        anns = []
        for part in self.parts:
            anns += all_anns[part]

        utils.colorful_print(f"--- num of episode: {len(anns)}", "green")

        steps = []
        for episode in tqdm(anns):
            action_list = []
            image_rpath_list = []
            for step_id, step in enumerate(episode):
                img_filename = step["img_filename"] + '.png'
                image_rpath = os.path.join(self.image_dir, img_filename)
                if not os.path.exists(image_rpath):
                    utils.colorful_print(f"{image_rpath} image not found", "red")
                    continue
                if len(img_filename) > 100:     
                    continue

                action_step, image_rpath = action2step(step, image_rpath)
                
                action_list.append(f"step {step_id}: {action_step} <image>")
                image_rpath_list.append(image_rpath)
            
            for step_id, step in enumerate(episode):
                steps.append({
                    "ep_id": step["ep_id"], 
                    "step_id": step_id,
                    "task": step["goal"], 
                    "action_list": action_list,
                    "image_path_list": image_rpath_list,
                    "action": action_list[step_id],
                    "image_path": image_rpath_list[step_id]
                })

        utils.colorful_print("--- Num of total step: " + str(len(steps)), "green")
        print(f"--- example\n{steps[:3]}")
        utils.write_json(steps, self.ann_wpath)

def post_precess_label_data(rpath, version):
    anns = utils.read_jsonl(rpath)
    train_anns = []

    for ann in anns:
        history_actions = "\n".join(ann["action_list"][:ann["step_id"]]).replace("<image>", "")
        if version == "v1":
            conversations = [
                {"role": "user", "content": system_prompt.format(ann["task"], history_actions, ann["action"])},
                {"role": "assistant", "content": str(ann["rating"])}
            ]
        else:
            conversations = [
                {"role": "user", "content": system_prompt.format(ann["task"], history_actions, ann["action"])},
                {"role": "assistant", "content": ann["response"]}
            ]
        image_path_list = [image_path.replace("\\", "/") for image_path in ann["image_path_list"]]
        train_anns.append({
            "ep_id": ann["ep_id"], 
            "step_id": ann["step_id"],
            "task": ann["task"], 
            "messages": conversations,
            "images": [image_path_list[ann["step_id"]]]
        })
        # print(conversations[0]["content"])
        # print(train_anns[0]["images"])
        # exit()
        
    
    utils.write_json(train_anns, wpath=rpath.split(".")[0] + f"_post_{version}.json")
    

# aitw_data = AITW(split="train", parts=["general", "webshopping", "install", "googleapps"], date="1022")
# aitw_data.get_label_data()

post_precess_label_data(rpath="data/aitw_anns/1022/aitw_train_label_v5_2k.jsonl", version="v2")