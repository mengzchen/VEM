import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from tqdm import tqdm
from typing import List

from data_preprocess import action2step, action_type_dict
import utils

system_prompt = """Task: Evaluate the prediction of Current action based on the given GUI screenshot and task requirements.

Instructions:
1. Review the history Action: Carefully examine the provided sequence of actions.
2. Assess the Task Requirements: Understand what the task is asking for.
3. Evaluate the Prediction of Current Action: Based on the provided entire actions and GUI screenshot, determine if current action aligns with the task requirements.
4. Rate the Action: Provide a rating based on four levels:
   - Level 1: The action is significantly deviating from the task requirements.
   - Level 2: The action contributes to the task requirements but may not be the most efficient path.
   - Level 3: The action is the optimal step towards achieving the task requirements.     

Explanation of Action:
- `action_type`: includes 'click', 'scroll down', 'scroll up', 'status task complete' (indicates the task is completed), 'press home' (return to the home page), 'press back', 'type' (typing text)
- `click_point`: if the action_type is click, this key will provide the relative position on the screenshot, (x, y) between [0, 1]
- `typed_text`: if the action_type is type, this key will provide the content.

Output Format:
[1-3]

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

                action_step, image_rpath = action2step(step, "aitw", image_rpath)
                
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


    def get_rl_data(self):
        all_anns = utils.read_json(self.ann_rpath)

        anns = []
        for part in self.parts:
            anns += all_anns[part]

        utils.colorful_print(f"--- num of episode: {len(anns)}", "green")

        steps = []
        for episode in tqdm(anns):
            action_list, autoui_action_list = [], []
            action_plans = []
            image_rpath_list = []
            for step_id, step in enumerate(episode):
                img_filename = step["img_filename"] + '.png'
                image_rpath = os.path.join(self.image_dir, img_filename)
                if not os.path.exists(image_rpath):
                    utils.colorful_print(f"{image_rpath} image not found", "red")
                    continue
                if len(img_filename) > 100:     
                    continue

                action_step, image_rpath = action2step(step, "aitw", image_rpath)
                
                action_list.append(f"step {step_id}: {action_step}")
                autoui_action_list.append(f"\"action_type\": \"{action_type_dict[step['action_type_id']]}\", \"touch_point\": \"{step['touch']}\", \"lift_point\": \"{step['lift']}\", \"typed_text\": \"{step['type_text']}\"")
                action_plans.append(action_type_dict[step['action_type_id']])
                image_rpath_list.append(image_rpath)
            

            for step_id, step in enumerate(episode):
                history_actions = "\n".join(action_list[:step_id]) 
                previous_actions = "\n".join(autoui_action_list[:step_id])
                image_path = image_rpath_list[step_id].replace("_modify", "").replace("jpg", "png")
                steps.append({
                    "ep_id": step["ep_id"], 
                    "step_id": step_id,
                    # "annot_position": step["annot_position"],
                    "image_path": image_path,
                    "critic_input": system_prompt.format(step["goal"], history_actions, f"Step {step_id}: ").rstrip("\n"),
                    "history_action": f"Previous Action:\n{previous_actions}\nGoal:\n{step['goal']}".replace("'", ""),
                    "next_action": f"Action Plan: {action_plans[step_id:]} ; Action Decision: {autoui_action_list[step_id]}".replace("'", "")
                })

        utils.colorful_print("--- Num of total step: " + str(len(steps)), "green")
        print(f"--- example\n{steps[:3]}")
        utils.write_json(steps, self.ann_wpath)


def combine_level(ann):
    if ann["rating"] == 3:
        ann["rating"] = 2
        ann["response"] = ann["response"].replace("Rating: 3", "Rating: 2")
    elif ann["rating"] == 4:
        ann["rating"] = 3
        ann["response"] = ann["response"].replace("Rating: 4", "Rating: 3")
    else:
        pass

    return ann


def post_precess_label_data(rpath, version):
    # todo 对训练数据做数据均衡
    anns = utils.read_jsonl(rpath)
    
    train_anns = []
    train_type_count = {1: 0, 2: 0, 3: 0, 4: 0}
    val_anns = []
    val_type_count = {1: 0, 2: 0, 3: 0, 4: 0}

    for ann in anns:
        ann = combine_level(ann)
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

        if ann["rating"] not in val_type_count.keys():
            pass
        elif val_type_count[ann["rating"]] < 50:
            val_anns.append({
                "ep_id": ann["ep_id"], 
                "step_id": ann["step_id"],
                "task": ann["task"], 
                "messages": conversations,
                "images": [image_path_list[ann["step_id"]]],
                "rating": ann["rating"]
            })
            val_type_count[ann["rating"]] += 1
        else:
            if ann["rating"] == 3:
                choice = np.random.choice(3, 1)
                if choice == 1:
                    train_anns.append({
                        "ep_id": ann["ep_id"], 
                        "step_id": ann["step_id"],
                        "task": ann["task"], 
                        "messages": conversations,
                        "images": [image_path_list[ann["step_id"]]],
                        "rating": ann["rating"]
                    })
                    train_type_count[ann["rating"]] += 1
            else:
                train_anns.append({
                    "ep_id": ann["ep_id"], 
                    "step_id": ann["step_id"],
                    "task": ann["task"], 
                    "messages": conversations,
                    "images": [image_path_list[ann["step_id"]]],
                    "rating": ann["rating"]
                })
                train_type_count[ann["rating"]] += 1
    
    utils.colorful_print(train_type_count, "green")
    utils.colorful_print(val_type_count, "green")

    utils.write_json(train_anns, wpath=rpath.split(".")[0] + f"_post_{version}.json")
    utils.write_json(val_anns, wpath=rpath.split(".")[0] + f"_post_{version}_val.json")
    

aitw_data = AITW(split="val", parts=["general"], date="1201")
aitw_data.get_rl_data()

# post_precess_label_data(rpath="data/aitw_anns/1102/aitw_train_label_v5.jsonl", version="v1")
