import os
import sys
sys.path.append(os.getcwd())

import google.generativeai as genai
import numpy as np
from PIL import Image
import re
from tqdm import tqdm
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
from data_preprocess.gpt import call_gemini


def autoui_translate_action(raw_action):
    pattern = r'(?<=Action Decision:\s).*'
    pred_str = "{" + re.search(pattern, raw_action).group(0).strip() + "}"
    step_data = ast.literal_eval(pred_str)
    action_type = str(step_data["action_type"])
    
    if action_type == "DUAL_POINT":
        touch_point = ast.literal_eval(step_data["touch_point"])
        lift_point = ast.literal_eval(step_data["lift_point"])
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point, lift_point=lift_point)
    elif action_type == "TYPE":
        typed_text = step_data["typed_text"] 
        return AndroidAction(action_type=ActionType.Type, typed_text=typed_text)
    elif action_type == 'PRESS_HOME':
        return AndroidAction(action_type=ActionType.GoHome)
    elif action_type == 'PRESS_BACK':
        return AndroidAction(action_type=ActionType.GoBack)
    elif action_type == 'PRESS_ENTER':
        return AndroidAction(action_type=ActionType.Enter)
    elif action_type == 'STATUS_TASK_COMPLETE':
        return AndroidAction(action_type=ActionType.TaskComplete)
    elif action_type == 'TASK_IMPOSSIBLE':
        return AndroidAction(action_type=ActionType.TaskImpossible)
    else:
        print(f"Action {raw_action} not supported yet.")
        return AndroidAction(action_type=ActionType.Idle)


def cogagent_translate_action(raw_action):
    raw_action = raw_action.split('Grounded Operation:')[1]
    action = raw_action.split(" ")[0]
    if action == 'tap':
        numbers = raw_action.split('[[')[1].split(',')
        x = int(numbers[0])
        y = int(numbers[1].split(']]')[0])
        touch_point = (x/1000, y/1000)
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point, lift_point=touch_point)
    elif "type" in action:
        text = raw_action.split('"')[1]
        return AndroidAction(action_type=ActionType.Type, typed_text=text)
    elif "press home" in raw_action:
        return AndroidAction(action_type=ActionType.GoHome)
    elif "press back" in raw_action:
        return AndroidAction(action_type=ActionType.GoBack)
    elif "press enter" in raw_action:
        return AndroidAction(action_type=ActionType.Enter)
    elif "task complete" in raw_action:
        return AndroidAction(action_type=ActionType.TaskComplete)
    elif "task impossible" in raw_action:
        return AndroidAction(action_type=ActionType.TaskImpossible)
    elif "swipe up" in raw_action:
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.5, 0.5), lift_point=(0.5, 0.2))
    elif "swipe down" in raw_action:
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.5, 0.2), lift_point=(0.5, 0.5))
    elif "swipe left" in raw_action:
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.8, 0.5), lift_point=(0.2, 0.5))
    elif "swipe right" in raw_action:
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.2, 0.5), lift_point=(0.8, 0.5))
    else:
        print(f"Action {raw_action} not supported yet.")
        return AndroidAction(action_type=ActionType.Idle)


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
        genai.configure(api_key=config["gemini_key"])
        self.client = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        self.threshold = 0.001 * 255**2


    def __call__(self, image_path, intent):
        system_msg, prompt, cot_image_list = build_prompt_general(intent)
        
        response_text = call_gemini(self.client, system_msg, prompt, cot_image_list, image_path)

        answer = re.search(r'Status:\s*(\w+)', response_text).group(1) if re.search(r'Status:\s*(\w+)', response_text) else None
        
        if answer is not None and "success" in answer.lower():
            return 1
        else:
            return 0
        

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
        self.max_steps = 10
        screen_size = self.driver.get_window_size()
        self.screen_size = (screen_size["width"], screen_size["height"])

        self.step_num, self.terminated = 0, False
        self.history = []
        if config["model_name"] == "cogagent":
            self.translate_action = cogagent_translate_action
        else:
            self.translate_action = autoui_translate_action
    

    def get_obs(self):
        screenshot_str = self.driver.get_screenshot_as_base64()
        imgdata = base64.b64decode(screenshot_str)
        image =  Image.open(BytesIO(imgdata))
        image_wpath = os.path.join(self.image_save_dir, f"{self.image_id}_{self.step_num}.png")
        image.save(image_wpath)
        
        return image_wpath


    def step(self, raw_action, task):
        action = self.translate_action(raw_action)
        self.history.append(action)
        self.step_num += 1
        if self.step_num > self.max_steps:
            action = AndroidAction(action_type=ActionType.TaskImpossible)
        
        if action.action_type == ActionType.DualPoint:
            assert len(action.touch_point) == 2
            assert len(action.lift_point) == 2
            touch_x = action.touch_point[0] * self.screen_size[0]
            touch_y = action.touch_point[1] * self.screen_size[1]
            lift_x = action.lift_point[0] * self.screen_size[0]
            lift_y = action.lift_point[1] * self.screen_size[1]
            if (touch_x - lift_x)**2 + (touch_y - lift_y)**2 < 10:
                self.driver.tap([(touch_x, touch_y)])
            else:
                self.driver.swipe(touch_x, touch_y, lift_x, lift_y)
        elif action.action_type == ActionType.Type:
            element = self.driver.switch_to.active_element
            element.send_keys(action.typed_text)
        elif action.action_type == ActionType.GoBack:
            self.driver.back()
        elif action.action_type == ActionType.GoHome:
            self.driver.press_keycode(3)
        elif action.action_type == ActionType.Enter:
            self.driver.press_keycode(66)
        elif action.action_type == ActionType.TaskComplete:
            self.terminated = True
        elif action.action_type == ActionType.TaskImpossible:
            self.terminated = True
        elif action.action_type == ActionType.Idle:
            pass
        else:
            raise Exception(f"Unknown action type: {action.action_type}")
        
        screenshot_path = self.get_obs()

        result = self.evaluator(
            screenshot_path,
            task
        )
        
        if result >= 1 or self.terminated:
            self.driver.quit()
        
        return screenshot_path, result


# env = AndroidEnv()
# env.step(raw_action="Action Decision: \"action_type\": \"DUAL_POINT\", \"touch_point\": \"[0.689, 0.778]\", \"lift_point\": \"[0.689, 0.778]\", \"typed_text\": \"\"")