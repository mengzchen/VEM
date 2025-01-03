import google.generativeai as genai
import numpy as np
from PIL import Image
import re
from tqdm import tqdm
import torch

from data_preprocess.prompt import build_prompt_general
from data_preprocess.gpt import call_gemini


class EndResultEvaluator:
    def __init__(self, config):
        genai.configure(api_key=config["gemini_key"])
        self.client = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        self.threshold = 0.001 * 255**2


    def __call__(self, last_two_images, intent):
        # TODO if need to get the two images
        with Image.open(last_two_images[0]) as img1_src, Image.open(last_two_images[1]) as img2_src:   
            img1 = np.array(img1_src)
            img2 = np.array(img2_src)
        if np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2) < self.threshold:
            print("skipping evaluation due to same images")
            return 0
        
        self.img_matrix = np.expand_dims(img2, axis = 0)
        
        eval_res = self._evaluate(intent, last_two_images[1])
            
        del img1, img2

        return eval_res

    def _evaluate(self, intent, image_path) -> bool:
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
    

    def reset(self):
        pass

    def step(self):
        pass


def interact_environment(agent, env, dataset):
    anns = []
    for data in tqdm(dataset):
        done = False
        ann = []
        
        observe = env.reset()
        
        screenshot = observe["image_feature"]
                
        steps = 0
        while not done:
            steps += 1
                
            action = agent.get_action(observe, screenshot)
            
            env_return = env.step(action)
            
            if env_return is None:
                done = True
                continue

            obs_dict, r, done = env_return
            next_screenshot = obs_dict["image_feature"]
            next_observe = obs_dict["prompt"]
            if not hasattr(agent, "critic"):
                ann.append({
                    "observation": observe, 
                    "next_observation": next_observe, 
                    "image_features": None, 
                    "image_path": obs_dict["image_path"], 
                    "next_image_features": None, 
                    "task": obs_dict["task"],
                    "reward": r, 
                    "done": done, 
                    "action": action})
                observe = obs_dict
            
            screenshot = next_screenshot
        
    anns.append(ann)  

    return anns
