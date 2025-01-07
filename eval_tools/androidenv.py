import os
import sys
sys.path.append(os.getcwd())

from PIL import Image
import re
from time import sleep

import base64
from io import BytesIO
import time

from appium import webdriver
from appium.options.android import UiAutomator2Options

from data_preprocess.prompt import build_prompt_general
from data_preprocess.cloudgpt_aoai import get_chat_completion, get_message
from data_preprocess.utils import autoui_translate_action, cogagent_translate_action
from data_preprocess.utils import ActionType


class EndResultEvaluator:
    def __init__(self):
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
        self.evaluator = EndResultEvaluator()

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
        action = self.translate_action(raw_action)
        for _ in range(2):
            try:
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
                    for i in range(2):
                        try:
                            sleep(4)
                            element = self.driver.switch_to.active_element
                            element.send_keys(action.typed_text)
                            break
                        except Exception as e:
                            print(f"The element is not loaded yet or agent did not click anything")
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
                break
            except Exception as e:
                sleep(10)
                continue
        
        sleep(5)
        screenshot_path = self.get_obs(step_num)

        done, explanation = self.evaluator(task, screenshot_path)
        
        return screenshot_path, done, action_description, grounded_operation, action, explanation


# env = AndroidEnv(config={"image_save_dir": "./images/online_eval_images", "appium_server_url": "http://10.150.140.70:4723", "model_name": "autogui"})
# raw_action = "Action Decision: \"action_type\": \"DUAL_POINT\", \"touch_point\": \"[0.689, 0.778]\", \"lift_point\": \"[0.689, 0.778]\", \"typed_text\": \"\""
# image_path, _, _, _, _, _ = env.step(raw_action=raw_action, task="Install facebook", step_num=0)
# print(image_path)