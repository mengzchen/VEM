import os
import sys
sys.path.append(os.getcwd())


import utils
from tqdm import tqdm
import json

from data_preprocess.cloudgpt_aoai import get_chat_completion, encode_image, get_message
from prompt import prompt_give_score, prompt_gen_1_action, prompt_gen_2_action
    

class GPTScorer:
    def __init__(self):
        self.prompt = prompt_give_score
    
    def get_label_anns(self, ann_rpath: str, ann_wpath: str):
        anns = utils.read_json(ann_rpath)
        uncomplete_anns = []
        if os.path.exists(ann_wpath):
            complete_anns = utils.read_jsonl(ann_wpath)
            print(f"--> complete anns: {len(complete_anns)}")
            complete_ids = [f"{ann['ep_id']}_{ann['step_id']}" for ann in complete_anns]
            for ann in anns:
                if f"{ann['ep_id']}_{ann['step_id']}" in complete_ids:
                    pass
                else:
                    uncomplete_anns.append(ann)
        else:
            uncomplete_anns = anns

        with open(ann_wpath, "+a") as fout:
            for ann in tqdm(uncomplete_anns):
                ann["response"] = self.get_one_answer(ann)
                ann["rating"] = utils.parse_rating(ann)
                fout.writelines(json.dumps(ann) + "\n")

        fout.close()


    def get_good_steps(self, ann_rpath):
        good_step_ids = []
        anns = utils.read_jsonl(ann_rpath)
        for ann in anns:
            if ann["rating"] == 3:
                good_step_ids.append(f"{ann['ep_id']}_{ann['step_id']}")

        return good_step_ids


    def get_negative_anns(self, ann_rpath: str, ann_wpath: str, num: int, level: int):
        score_path = ann_rpath.replace(".json", ".jsonl").replace("label", "score")
        
        good_step_ids = self.get_good_steps(score_path)[:num]
        anns = utils.read_json(ann_rpath)
        anns = [ann for ann in anns if f"{ann['ep_id']}_{ann['step_id']}" in good_step_ids]

        # TODO modify image path
        for ann in anns:
            ann["image_path"] = ann["image_path"].replace("data/", "")
            ann["image_path_list"] = [image_path.replace("data/", "") for image_path in ann["image_path_list"]]

        uncomplete_anns = []
        if os.path.exists(ann_wpath):
            complete_anns = utils.read_jsonl(ann_wpath)
            print(f"--> complete anns: {len(complete_anns)}")
            complete_ids = [f"{ann['ep_id']}_{ann['step_id']}" for ann in complete_anns]
            for ann in anns:
                if f"{ann['ep_id']}_{ann['step_id']}" in complete_ids:
                    pass
                else:
                    uncomplete_anns.append(ann)
        else:
            uncomplete_anns = anns

        if level == 1:
            self.prompt = prompt_gen_1_action
        else:
            self.prompt = prompt_gen_2_action

        with open(ann_wpath, "+a") as fout:
            for ann in tqdm(uncomplete_anns):
                new_action = self.get_one_answer(ann)
                print(new_action)
                ann["action"] = f"step {ann['step_id']}: {new_action} <image>"
                ann["rating"] = level
                fout.writelines(json.dumps(ann) + "\n")

        fout.close()


    def get_one_answer(self, step: dict) -> str:
        other = """
Task Requirements: {}
Action and ScreenShot:
{}
Origin Action:
{}
        """
        task_describe = self.prompt + other.format(step["task"], "\n".join(step["action_list"]), step["action"])
        message = get_message(task_describe.split("<image>")[:-1], step["image_path_list"] + [step["image_path"]])
        
        response = get_chat_completion(
            engine="gpt-4o-20240513",
            messages=message,
        )

        return response.choices[0].message.content


ann_rpath = "data/aitw_anns/1218/aitw_train_label.json"
ann_wpath = "data/aitw_anns/1218/aitw_train_score_1.jsonl"
# GPTScorer().get_label_anns(ann_rpath=ann_rpath, ann_wpath=ann_wpath)
# GPTScorer().get_negative_anns(ann_rpath=ann_rpath, ann_wpath=ann_wpath, num=500, level=1)
GPTScorer().get_negative_anns(ann_rpath=ann_rpath, ann_wpath=ann_wpath, num=100, level=1)