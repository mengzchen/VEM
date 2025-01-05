import argparse
import yaml
import json
import os
from tqdm import tqdm

import utils
from dataset import create_dataset
from models import create_agent
from eval_tools.androidenv import AndroidEnv 



def evaluation(agent, dataset, env):
    anns = []
    for (task, query_format) in tqdm(dataset):
        done, history = False, []
        ann = []
        
        screenshot_path = env.get_obs()
        step_num = 0
        while not done:
            step_num += 1
            text = query_format.format(task, "".join(history))
            raw_action = agent.get_action(text=text, image_path=screenshot_path)
            print(raw_action)
            
            image = env.step(raw_action)
            
            exit()

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


def main(config):
    print("config:", json.dumps(config))
    output_path = os.path.join("checkpoints/results/", f"{config['model_name']}.jsonl")
    print("output_path: ", output_path)

    print("### build android env")

    env = AndroidEnv(config)

    print("### Creating agent")
    agent = create_agent(config)
    
    print("### Creating datasets")
    dataset = create_dataset(config)

    print("### Start evaluating")

    predictions = evaluation(agent, dataset, env)

    utils.write_jsonl(predictions, output_path)
    print("### Prediction Results Save To: ", output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default="")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)