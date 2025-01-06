import argparse
import yaml
import json
import os
from tqdm import tqdm

import utils
from dataset import create_dataset
from models import create_agent
from eval_tools.androidenv import AndroidEnv, ActionType
from data_preprocess.action_transfer import step_2_action



def evaluation(config, agent, dataset, env, ann_wpath):
    with open(ann_wpath, "a") as fout:
        for task_id, (task, query_format) in enumerate(dataset):
            done, history = False, []

            step_num = 0
            screenshot_path = env.get_obs(step_num)
            while not done:
                step_num += 1
                text = query_format.format(task, "".join(history))
                raw_action = agent.get_action(text=text, image_path=screenshot_path)
                
                screenshot_path, done, action_description, grounded_operation, action, explanation = env.step(raw_action, task, step_num)
                
                print(f"======\n{task_id}_{step_num}: {task}\nimage: {screenshot_path}\ntext: {text}\ngrounding: {grounded_operation}\naction type: {action.action_type}\ndone: {done}\ngpt: {explanation}\ndesc: {action_description}")
                if config["model_name"] == "cogagent":
                    history.append(f"\n{step_num-1}. {grounded_operation}\t{action_description}")
                elif config["model_name"] == "autogui":
                    action_desc = step_2_action(
                        action_type=action.action_type, 
                        touch_point=action.touch_point,
                        lift_point=action.lift_point,
                        typed_text=action.type_text,
                        add_all_dict=True
                    )
                    history.append(action_desc)
                else:
                    raise KeyError
                
                result ={
                    "task_id": task_id,
                    "step_id": step_num,
                    "task": task,
                    "image_path": screenshot_path.replace("\\", "/"),
                    "if_done": done,
                    "prompt": text,
                    "gpt-4o": explanation
                }
                fout.writelines(json.dumps(result) + "\n")

                if step_num > 10 or done or action.action_type == ActionType.TaskComplete:
                    env.driver.press_keycode(3)
                    break
                
    fout.close()


def main(config):
    print("config:", json.dumps(config))
    ann_wpath = os.path.join("data", f"{config['model_name']}_online_aitw_general.jsonl")
    finish_task = [ann["task"] for ann in utils.read_jsonl(ann_wpath)]
    success_num = sum([ann["if_done"] for ann in utils.read_jsonl(ann_wpath)])
    print(f"### finish task num: {len(finish_task)}\tsuccess: {success_num}")
    print("output_path: ", ann_wpath)

    print("- build android env")

    env = AndroidEnv(config)

    print("- Creating agent")
    agent = create_agent(config)
    
    print("- Creating datasets")
    dataset = create_dataset(config, finish_task)

    print("### Start evaluating")

    evaluation(config, agent, dataset, env, ann_wpath)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default="")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)