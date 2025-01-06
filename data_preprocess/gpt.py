import os
import sys
sys.path.append(os.getcwd())

import io
from PIL import Image

from data_preprocess.cloudgpt_aoai import get_chat_completion, get_message
from data_preprocess.prompt import prompt_score_system, prompt_score_user
    

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


def process_image(image_path):
    image = Image.open(image_path, 'r')
    image = image.resize((image.width // 4, image.height // 4))
    # Save to a BytesIO object (in-memory file) as PNG
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    # Load it back from the BytesIO object
    buffer.seek(0)
    image_reloaded = Image.open(buffer)
    return image_reloaded


def call_gemini(client, system_msg, prompt, image_list, image_path):
    input_msg = [system_msg + "\n" + "=====Examples====="]
    for i in range(len(image_list)-1):
        input_msg += [
            "\nScreenshot:",
            process_image(image_list[i]),
            prompt[i]
        ]
    input_msg += [
        "=====Your Turn=====",
        "\nScreenshot: ",
        process_image(image_path),
        prompt[-1]
    ]
    
    response = client.generate_content(
        input_msg
    )

    response.resolve()
    response_text = response.text

    return response_text


# from data_preprocess.prompt import test

# message = get_message(test, [])
        
# response = get_chat_completion(
#     engine="gpt-4o-20240513",
#     messages=message,
# )

# print(response.choices[0].message.content)