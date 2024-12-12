import os
import sys
sys.path.append(os.getcwd())


import utils
from tqdm import tqdm
import json

from data_preprocess.cloudgpt_aoai import get_chat_completion, encode_image, get_message
from prompt import prompt_give_score
    

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


    def get_one_answer(self, step: dict) -> str:
        task_describe = self.prompt.format(step["task"], "\n".join(step["action_list"]), step["action"])
        message = get_message(task_describe.split("<image>")[:-1], step["image_path_list"] + [step["image_path"]])
        
        response = get_chat_completion(
            engine="gpt-4o-20240513",
            messages=message,
        )

        return response.choices[0].message.content


ann_rpath = "data/aitw_anns/1209/aitw_val_label.json"
ann_wpath = "data/aitw_anns/1209/aitw_val_score.jsonl"
GPTScorer().get_label_anns(ann_rpath=ann_rpath, ann_wpath=ann_wpath)

