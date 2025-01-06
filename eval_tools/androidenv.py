import os
import sys
sys.path.append(os.getcwd())

import google.generativeai as genai
from PIL import Image
import re
from time import sleep
from enum import Enum
from typing import Tuple
import base64
from io import BytesIO
import ast
from dataclasses import dataclass
import time

from appium import webdriver
from appium.options.android import UiAutomator2Options

from data_preprocess.prompt import build_prompt_general
from data_preprocess.cloudgpt_aoai import get_chat_completion, get_message
from eval_tools.aitw import str_2_format


def autoui_translate_action(raw_action):
    action_desc = str_2_format(raw_action)
    pattern = r'(?<=Action Decision:\s).*'
    pred_str = "{" + re.search(pattern, raw_action).group(0).strip() + "}"
    step_data = ast.literal_eval(pred_str)
    action_type = str(step_data["action_type"])
    
    if action_type == "DUAL_POINT":
        touch_point = ast.literal_eval(step_data["touch_point"])
        lift_point = ast.literal_eval(step_data["lift_point"])
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point, lift_point=lift_point), action_desc, None
    elif action_type == "TYPE":
        typed_text = step_data["typed_text"] 
        return AndroidAction(action_type=ActionType.Type, typed_text=typed_text), action_desc, None
    elif action_type == 'PRESS_HOME':
        return AndroidAction(action_type=ActionType.GoHome), action_desc, None
    elif action_type == 'PRESS_BACK':
        return AndroidAction(action_type=ActionType.GoBack), action_desc, None
    elif action_type == 'PRESS_ENTER':
        return AndroidAction(action_type=ActionType.Enter), action_desc, None
    elif action_type == 'STATUS_TASK_COMPLETE':
        return AndroidAction(action_type=ActionType.TaskComplete), action_desc, None
    elif action_type == 'TASK_IMPOSSIBLE':
        return AndroidAction(action_type=ActionType.TaskImpossible), action_desc, None
    else:
        print(f"Action {raw_action} not supported yet.")
        return AndroidAction(action_type=ActionType.Idle), action_desc, None


def cogagent_translate_action(raw_action):
    grounded_operation, action_description = None, None
    bounding_boxes = None

    grounded_operation_pattern = r"Grounded Operation:\s*(.*)"
    action_description_pattern = r"Action:\s*(.*)"
    matches_grounded_operation = re.search(grounded_operation_pattern, raw_action)
    matches_action_description = re.search(action_description_pattern, raw_action)

    if matches_grounded_operation:
        grounded_operation = matches_grounded_operation.group(1)
    if matches_action_description:
        action_description = matches_action_description.group(1)

    bounding_boxes_pattern = r"box=\[\[?(\d+),(\d+),(\d+),(\d+)\]?\]"
    matches_bounding_boxes = re.findall(bounding_boxes_pattern, raw_action)
    if matches_bounding_boxes:
        bounding_boxes = [[int(x) / 1000 for x in match] for match in matches_bounding_boxes][0]

    action_class = None
    if "CLICK" in grounded_operation:
        touch_point = (bounding_boxes[0], bounding_boxes[1])
        action_class = AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point, lift_point=touch_point)
    elif "TYPE" in grounded_operation:
        match_text = re.search(r"text='([^']*)'", grounded_operation)
        if match_text:
            text = match_text.group(1)
        else:
            text = ""
        action_class = AndroidAction(action_type=ActionType.Type, typed_text=text)
    elif "PRESS_HOME" in grounded_operation:
        action_class = AndroidAction(action_type=ActionType.GoHome)
    elif "PRESS_BACK" in grounded_operation:
        action_class = AndroidAction(action_type=ActionType.GoBack)
    elif "PRESS_ENTER" in grounded_operation:
        action_class = AndroidAction(action_type=ActionType.Enter)
    elif "END" in grounded_operation:
        action_class = AndroidAction(action_type=ActionType.TaskComplete)
    elif "SCROLL_UP" in grounded_operation:
        action_class = AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.5, 0.5), lift_point=(0.5, 0.2))
    elif "SCROLL_DOWN" in grounded_operation:
        action_class = AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.5, 0.2), lift_point=(0.5, 0.5))
    elif "SCROLL_LEFT" in grounded_operation:
        action_class = AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.8, 0.5), lift_point=(0.2, 0.5))
    elif "SCROLL_RIGHT" in grounded_operation:
        action_class = AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.2, 0.5), lift_point=(0.8, 0.5))
    else:
        print(f"Action {raw_action} not supported yet.")
        action_class = AndroidAction(action_type=ActionType.Idle)

    return action_class, action_description, grounded_operation


class ActionType(Enum):
    Idle=0
    DualPoint=1
    Type=2
    GoBack=3
    GoHome=4
    Enter=5
    TaskComplete=6
    TaskImpossible=7


@dataclass
class AndroidAction():
    action_type: ActionType
    touch_point: Tuple[float, float] = None
    lift_point: Tuple[float, float] = None
    typed_text: str = None


class EndResultEvaluator:
    def __init__(self, config):
        pass


    def __call__(self, task, image_path):
        prompt, image_list = build_prompt_general(task, image_path)
        message = get_message(prompt, image_list)
        
        response = get_chat_completion(
            engine="gpt-4o-20240513",
            messages=message,
        ).choices[0].message.content
        
        answer = re.search(r'Status:\s*(\w+)', response).group(1) if re.search(r'Status:\s*(\w+)', response) else None
        
        if answer is not None and "success" in answer.lower():
            return 1, response
        else:
            return 0, response
        

class AndroidEnv:
    def __init__(self, config):
        self.evaluator = EndResultEvaluator(config)

        self.image_save_dir = config["image_save_dir"]
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)
        self.image_id = str(time.time())

        self.appium_server_url = config["appium_server_url"]
        capabilities = dict(
            platformName='Android',
            automationName='uiautomator2',
            deviceName='Android',
            newCommandTimeout="120000",
            adbExecTimeout="120000",
            uiautomator2ServerInstallTimeout="120000",
            uiautomator2ServerLaunchTimeout="120000",
            uiautomator2ServerReadTimeout="120000",
            noSign=True
        )
        self.options = UiAutomator2Options().load_capabilities(capabilities)
        self.driver = webdriver.Remote(self.appium_server_url, options=self.options)
        screen_size = self.driver.get_window_size()
        self.screen_size = (screen_size["width"], screen_size["height"])

        if config["model_name"] == "cogagent":
            self.translate_action = cogagent_translate_action
        else:
            self.translate_action = autoui_translate_action
    

    def get_obs(self, step_num):
        screenshot_str = self.driver.get_screenshot_as_base64()
        imgdata = base64.b64decode(screenshot_str)
        image =  Image.open(BytesIO(imgdata))
        image_wpath = os.path.join(self.image_save_dir, f"{self.image_id}_{step_num}.png")
        image.save(image_wpath)
        
        return image_wpath


    def step(self, raw_action, task, step_num):
        action, action_description, grounded_operation = self.translate_action(raw_action)
        
        if action.action_type == ActionType.DualPoint:
            assert len(action.touch_point) == 2
            assert len(action.lift_point) == 2
            touch_x = action.touch_point[0] * self.screen_size[0]
            touch_y = action.touch_point[1] * self.screen_size[1]
            lift_x = action.lift_point[0] * self.screen_size[0]
            lift_y = action.lift_point[1] * self.screen_size[1]
            if (touch_x - lift_x)**2 + (touch_y - lift_y)**2 == 0:
                self.driver.tap([(touch_x, touch_y)])
            else:
                self.driver.swipe(touch_x, touch_y, lift_x, lift_y)
        elif action.action_type == ActionType.Type:
            for _ in range(2):
                try:
                    sleep(4)
                    element = self.driver.switch_to.active_element
                    element.send_keys(action.typed_text)
                    break
                except Exception as e:
                    print("The element is not loaded yet or agent did not click anything")
        elif action.action_type == ActionType.GoBack:
            self.driver.back()
        elif action.action_type == ActionType.GoHome:
            self.driver.press_keycode(3)
        elif action.action_type == ActionType.Enter:
            self.driver.press_keycode(66)
        elif action.action_type == ActionType.TaskComplete:
            # self.driver.press_keycode(3)
            pass
        else:
            pass
        
        screenshot_path = self.get_obs(step_num)

        done, explanation = self.evaluator(task, screenshot_path)
        
        return screenshot_path, done, action_description, grounded_operation, action, explanation


# env = AndroidEnv()
# env.step(raw_action="Action Decision: \"action_type\": \"DUAL_POINT\", \"touch_point\": \"[0.689, 0.778]\", \"lift_point\": \"[0.689, 0.778]\", \"typed_text\": \"\"")