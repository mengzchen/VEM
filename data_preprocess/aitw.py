import os
import sys
sys.path.append(os.getcwd())

from tqdm import tqdm
from typing import List

import utils


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
        self.ann_rpath = f"data/aitw_data_{split}.json"
        # choice: general single webshopping install googleapps
        self.parts = parts
        self.ann_wpath = f"data/aitw_{split}_{date}.json"
    
    def get_label_data(self):
        all_anns = utils.read_json(self.ann_rpath)

        anns = []
        for part in self.parts:
            anns += all_anns[part]
        steps = []
        utils.colorful_print(f"--- num of episode: {len(anns)}", "green")

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

aitw_data = AITW(split="train", parts=["general"], date="1017")
aitw_data.get_label_data()