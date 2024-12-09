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

        print(f"--- num of episode: {len(anns)}")

        steps = []
        for episode in tqdm(anns):
            action_list = []
            image_rpath_list = []
            for step_id, step in enumerate(episode):
                image_filename = f"{step['img_filename']}.png"
                image_rpath = os.path.join(self.image_dir, image_filename)
                if not os.path.exists(image_rpath):
                    print(f"{image_rpath} image not found")
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

        print("--- Num of total step: " + str(len(steps)))
        print(f"--- example\n{steps[:3]}")
        utils.write_json(steps, self.ann_wpath)


    def get_negative_label_data(self):
        """
        TODO use GPT to generate negative data, ask it to give 1 or 2 score action
        sample some data from general task
        """
        all_anns = utils.read_json(self.ann_wpath)


    def get_rl_data(self):
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

                qwen2vl_action_step, image_rpath = action2step(step, "aitw", image_rpath)
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
    anns = utils.read_jsonl(rpath)
    
    train_anns = []
    train_type_count = {1: 0, 2: 0, 3: 0, 4: 0}
    val_anns = []
    val_type_count = {1: 0, 2: 0, 3: 0, 4: 0}

    for ann in anns:
        ann = combine_level(ann)
        history_actions = "\n".join(ann["action_list"][:ann["step_id"]]).replace("<image>", "")

        conversations = [
            {"role": "user", "content": system_prompt.format(ann["task"], history_actions, ann["action"])},
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
        if ann["rating"] not in val_type_count.keys():
            pass
        elif val_type_count[ann["rating"]] < 50:
            val_anns.append(example)
            val_type_count[ann["rating"]] += 1
        else:
            if ann["rating"] == 3:
                choice = np.random.choice(3, 1)
                if choice == 1:
                    train_anns.append(example)
                    train_type_count[ann["rating"]] += 1
            else:
                train_anns.append(example)
                train_type_count[ann["rating"]] += 1
    
    print(train_type_count)
    print(val_type_count)

    utils.write_json(train_anns, wpath=rpath.split(".")[0] + f"_post_{version}.json")
    utils.write_json(val_anns, wpath=rpath.split(".")[0] + f"_post_{version}_val.json")
    

aitw_data = AITW(split="val", parts=["general"], date="1201")
aitw_data.get_rl_data()

# post_precess_label_data(rpath="data/aitw_anns/1102/aitw_train_label_v5.jsonl", version="v1")
