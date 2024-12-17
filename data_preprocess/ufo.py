import os
import sys
sys.path.append(os.getcwd())

from tqdm import tqdm
import random
from data_preprocess import action2step

import utils


class UFO:
    def __init__(self, dataset_name:str, split: str, date: str):
        self.dataset_name = dataset_name
        self.image_dir = f"images/{dataset_name}_images"
        self.split = split
        self.train_ids = utils.read_json("data/ufo_anns/split.json")["train_ids"]
        self.test_ids = utils.read_json("data/ufo_anns/split.json")["test_ids"]
        
        if not os.path.exists(f"data/{dataset_name}_anns/{date}"):
            os.mkdir(f"data/{dataset_name}_anns/{date}")
        self.ann_wpath = f"data/{dataset_name}_anns/{date}/{dataset_name}_{split}.json"
    
    def get_label_data(self):
        if self.split == "train":
            ids = self.train_ids
        else:
            ids = self.test_ids

        utils.colorful_print(f"--- num of episode: {len(ids)}", "green")

        steps = []
        for id in tqdm(ids):
            ann_path = f"data/ufo_anns/ufo_origin_data/{id}.jsonl"
            ann = utils.read_jsonl(ann_path)
            
            for step in ann:
                step_id = step["step_id"]
                image_rpath = os.path.join(self.image_dir, id, f"{step_id}_annotated.png")
                if not os.path.exists(image_rpath):
                    utils.colorful_print(f"{image_rpath} image not found", "red")
                    continue
                
                image_rpath = image_rpath.replace("\\", "/")
                
                # add user screenshot
                conversations = [
                    {"role": "user", "content": f"{step['input']['system']}\n{step['input']['user']}\n<Current Screenshot>: <image>"},
                    {"role": "assistant", "content": str(step["output"])}
                ]
                
                steps.append({
                    "messages": conversations,
                    "images": [image_rpath]
                })

        utils.colorful_print("--- Num of total step: " + str(len(steps)), "green")
        
        random.shuffle(steps)
        utils.write_json(steps, self.ann_wpath)

ufo_data = UFO(dataset_name="ufo", split="test", date="1105")
ufo_data.get_label_data()
