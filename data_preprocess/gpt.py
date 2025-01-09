import os
import sys
sys.path.append(os.getcwd())

import io
from PIL import Image

import utils
from data_preprocess.cloudgpt_aoai import get_chat_completion, get_message
from data_preprocess.prompt import prompt_score_system, prompt_score_user, prompt_negative_system, prompt_negative_user
from data_preprocess.utils import action_dict_to_class, to_autoui


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
            
        task_describe = prompt_negative_system + prompt_negative_user.format(task, history_action_desc, f"\n<image>\n{ann['action_desc_list'][ann['step_id']]}")
        
        message = get_message(task_describe.split("<image>")[:-1], ann["add_point_image_list"] + [ann["add_point_image_list"][ann["step_id"]]])
        
        response = get_chat_completion(
            engine="gpt-4o-20240513",
            messages=message,
        ).choices[0].message.content.replace("```json", "").replace("```", "")
        
        action_desc = eval(response)["action_desc"]
        if "click_point" in action_desc.keys():
            click_point = [float(element.strip()) for element in action_desc["click_point"][1:-1].split(',')]
            action_desc["touch_point"], action_desc["lift_point"] = click_point, click_point
            del action_desc["click_point"]

        add_point_image_path = utils.add_visilize2screenshot(ann["add_point_image_list"][ann["step_id"]], action_desc, "negative")
        
        return to_autoui(action_dict_to_class(action_desc), all_dict=False), add_point_image_path

def process_image(image_path):
    image = Image.open(image_path, 'r')
    image = image.resize((image.width // 4, image.height // 4))
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    buffer.seek(0)
    image_reloaded = Image.open(buffer)
    return image_reloaded


if __name__ == "__main__":
    pass
    # from data_preprocess.prompt import test

    # message = get_message(test, [])
            
    # response = get_chat_completion(
    #     engine="gpt-4o-20240513",
    #     messages=message,
    # )

    # print(response.choices[0].message.content)