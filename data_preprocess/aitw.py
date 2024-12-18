import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from tqdm import tqdm
from typing import List

from data_preprocess.action_transfer import action2step, action_type_dict, aitw_step_update
import utils
from data_preprocess.prompt import prompt_critic_input


class AITW:
    def __init__(self, split: str, parts: List, date: str):
        self.image_dir = "images/aitw_images"
        self.split = split
        self.ann_rpath = f"data/aitw_anns/aitw_{split}.json"

        # choice: general single webshopping install googleapps
        self.parts = parts
        if not os.path.exists(f"data/aitw_anns/{date}"):
            os.mkdir(f"data/aitw_anns/{date}")
        self.date = date
    

    def get_label_data(self):
        all_anns = utils.read_json(self.ann_rpath)
        ann_wpath = f"data/aitw_anns/{self.date}/aitw_{self.split}_label.json"

        anns = []
        for part in self.parts:
            anns += all_anns[part]

        print(f"--- num of episode: {len(anns)}")

        steps = []
        for episode in tqdm(anns):
            action_list, image_path_list = [], []
            for step_id, step in enumerate(episode):
                image_filename = f"{step['img_filename']}.png"
                image_path = os.path.join(self.image_dir, image_filename)
                if not os.path.exists(image_path):
                    print(f"{image_path} image not found")
                    continue

                action_step, image_path = action2step(step, "aitw", image_path, add_visual=True)
                action_step = f"\"action_type\": \"{action_type_dict[step['action_type_text']]}\", \"touch_point\": \"{step['touch']}\", \"lift_point\": \"{step['lift']}\", \"typed_text\": \"{step['type_text']}\" <image>"
                action_list.append(f"step {step_id}: {action_step}")
                image_path_list.append(image_path)
            
            for step_id, step in enumerate(episode):
                steps.append({
                    "ep_id": step["ep_id"], 
                    "step_id": step_id,
                    "task": step["goal"], 
                    "action_list": action_list,
                    "image_path_list": image_path_list,
                    "action": action_list[step_id],
                    "image_path": image_path_list[step_id]
                })

        print(f"--- example\n{steps[:2]}")
        utils.write_json(steps, ann_wpath)
        print("--- Num of total step: " + str(len(steps)))


    def get_rl_data(self):
        ann_wpath = f"data/aitw_anns/{self.date}/aitw_{self.split}_policy.json"
        all_anns = utils.read_json(self.ann_rpath)

        anns = []
        for part in self.parts:
            anns += all_anns[part]

        print(f"--- num of episode: {len(anns)}")

        steps = []
        for episode in tqdm(anns):
            qwen2vl_action_list, image_rpath_list = [], []
            autogui_action_list, autogui_action_plans = [], []

            for step_id, step in enumerate(episode):
                image_rpath = os.path.join(self.image_dir, f"{step['img_filename']}.png")
                assert os.path.exists(image_rpath), f"{image_rpath} image not found"

                qwen2vl_action_step, image_rpath = action2step(step, "aitw", image_rpath, add_visual=False)
                step = aitw_step_update(step)

                qwen2vl_action_list.append(f"step {step_id}: {qwen2vl_action_step}")
                autogui_action_list.append(f"\"action_type\": \"{action_type_dict[step['action_type_text']]}\", \"touch_point\": \"{step['touch']}\", \"lift_point\": \"{step['lift']}\", \"typed_text\": \"{step['type_text']}\"")
                autogui_action_plans.append(action_type_dict[step['action_type_text']])
                image_rpath_list.append(image_rpath)
            

            for step_id, step in enumerate(episode):
                history_actions = "\n".join(qwen2vl_action_list[:step_id])
                previous_actions = "\n".join(autogui_action_list[:step_id])
                image_path = image_rpath_list[step_id].replace("_modify", "").replace("jpg", "png")
                steps.append({
                    "ep_id": step["ep_id"], 
                    "step_id": step_id,
                    "image_path": image_path,
                    "critic_input": prompt_critic_input.format(step["goal"], history_actions, f"Step {step_id}: ").rstrip("\n"),
                    "history_action": f"Previous Action:\n{previous_actions}\nGoal:\n{step['goal']}".replace("'", ""),
                    "next_action": f"Action Plan: {autogui_action_plans[step_id:]} ; Action Decision: {autogui_action_list[step_id]}".replace("'", "")
                })

        print("--- Num of total step: " + str(len(steps)))
        print(f"--- example\n{steps[:3]}")
        utils.write_json(steps, ann_wpath)


# aitw use general critic => webshopping cross-task critic
def post_precess_label_data(rpath):
    anns = utils.read_jsonl(rpath)
    
    train_anns = []
    type_count = {1: 0, 2: 0, 3: 0}

    for ann in anns:
        ann["image_path"] = ann["image_path"].replace("data/", "")
        ann["image_path_list"] = [image_path.replace("data/", "") for image_path in ann["image_path_list"]]
        assert os.path.exists(ann["image_path"]), f"{ann['image_path']} not found"

        history_actions = "\n".join(ann["action_list"][:ann["step_id"]]).replace("<image>", "")

        conversations = [
            {"role": "user", "content": prompt_critic_input.format(ann["task"], history_actions, ann["action"].replace("<image>", ""))},
            {"role": "assistant", "content": str(ann["rating"])}
        ]
        
        image_path_list = [image_path.replace("\\", "/") for image_path in ann["image_path_list"]]

        example = {
            "ep_id": ann["ep_id"],
            "step_id": ann["step_id"],
            "task": ann["task"],
            "messages": conversations,
            "images": [image_path_list[ann["step_id"]]],
            "rating": ann["rating"]
        }

        train_anns.append(example)
        type_count[ann["rating"]] += 1
    
    print(type_count)

    utils.write_json(train_anns, wpath=rpath.split(".")[0].replace("_score", "") + f"_critic.json")
    

aitw_data = AITW(split="val", parts=["general"], date="1209")
aitw_data.get_rl_data()
# aitw_data.get_label_data()

# post_precess_label_data(rpath="data/aitw_anns/1209/aitw_train_score_2.jsonl")
