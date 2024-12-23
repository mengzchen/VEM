import os
import sys
sys.path.append(os.getcwd())


import utils
from tqdm import tqdm
import json

from data_preprocess.cloudgpt_aoai import get_chat_completion, get_message
from prompt import prompt_score_system, prompt_score_user, prompt_negative_system, prompt_negative_user
from data_preprocess.action_transfer import step_2_action
    

class GPTScorer:
    def __init__(self):
        pass


    def get_negative_anns(self, ann_rpath: str, ann_wpath: str, num: int, level: int):
        pass


    def get_score(self, ann):
        task = ann["task"]

        # add <image> token to prompt
        history_action_desc = ""
        for action_desc in ann["action_desc_list"]:
            history_action_desc += f"\n<image>\n{action_desc}"
        task_describe = prompt_score_system + prompt_score_user.format(task, history_action_desc, f"\n<image>\n{ann['action_desc_list'][ann['step_id']]}")
        
        message = get_message(task_describe.split("<image>")[:-1], ann["add_point_image_list"] + [ann["add_point_image_list"][ann["step_id"]]])
        
        response = get_chat_completion(
            engine="gpt-4o-20240513",
            messages=message,
        )
        
        return response.choices[0].message.content

    def get_negative_action(self, ann):
        task = ann["task"]

        history_action_desc = ""
        for action_desc in ann["action_desc_list"]:
            history_action_desc += f"\n<image>\n{action_desc}"
        task_describe = prompt_score_system + prompt_score_user.format(task, history_action_desc, f"\n<image>\n{ann['action_desc_list'][ann['step_id']]}")
        
        message = get_message(task_describe.split("<image>")[:-1], ann["add_point_image_list"] + [ann["add_point_image_list"][ann["step_id"]]])
        
        response = get_chat_completion(
            engine="gpt-4o-20240513",
            messages=message,
        )
        
        return response.choices[0].message.content


# from data_preprocess.prompt import test

# message = get_message(test, [])
        
# response = get_chat_completion(
#     engine="gpt-4o-20240513",
#     messages=message,
# )

# print(response.choices[0].message.content)