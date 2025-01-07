import argparse
import yaml
import json
import os
import cv2

import utils
from dataset import create_dataset
from models import create_agent
from eval_tools.androidenv import AndroidEnv, ActionType
from eval_tools.androidenv import to_autoui


def add_visilize2screenshot(image_rpath, action):
    if action.action_type != ActionType.DualPoint:
        return image_rpath

    image = cv2.imread(image_rpath)
    height, width, _ = image.shape

    x = int(action.touch_point[0] * width)
    y = int(action.touch_point[1] * height)

    cv2.circle(image, (x, y), 20, (0, 0, 255), -1)

    image_wpath = image_rpath.replace(".png", "") + f"_point.png"
    cv2.imwrite(image_wpath, image) 

    return image_wpath


def evaluation(config, agent, dataset, env, ann_wpath):
    with open(ann_wpath, "a") as fout:
        for task_id, (task, query_format) in enumerate(dataset):
            done, history = False, []

            step_num = 0
            screenshot_path = env.get_obs(step_num)
            while not done:
                print(f"============\n{task_id}_{step_num}: {task}\ncurrent_image: {screenshot_path}")
                step_num += 1
                if config["model_name"] == "cogagent":
                    text = query_format.format(task, "".join(history))
                else:
                    text = query_format.format("".join(history), task)
                
                raw_action = agent.get_action(text=text, image_path=screenshot_path)
                action, _, _ = env.translate_action(raw_action)
                point_image_path = add_visilize2screenshot(screenshot_path, action)
                
                screenshot_path, done, action_description, grounded_operation, action, explanation = env.step(raw_action, task, step_num)

                print(f"action: {action}\nafter image: {point_image_path}\nif done: {done}\ndesc: {action_description}")

                if config["model_name"] == "cogagent":
                    history.append(f"\n{step_num-1}. {grounded_operation}\t{action_description}")
                elif config["model_name"] == "autogui":
                    action_desc = to_autoui(action)
                    history.append(action_desc)
                else:
                    raise KeyError
                
                result ={
                    "task_id": task_id,
                    "step_id": step_num,
                    "task": task,
                    "image_path": screenshot_path.replace("\\", "/"),
                    "point_image_path": point_image_path,
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
    ann_wpath = f"./data/{config['output_name']}_online_aitw_general.jsonl"
    finish_task, success_num, step_len = {}, 0, 0
    if os.path.exists(ann_wpath):
        for ann in utils.read_jsonl(ann_wpath):
            if ann["task"] not in finish_task:
                finish_task[ann["task"]] = {"success": False, "steps": []}
            if ann["if_done"]:
                finish_task[ann["task"]]["success"] = True
            finish_task[ann["task"]]["steps"].append(ann)

        for task, info in finish_task.items():
            if info["success"]: success_num += 1
            step_len += len(info["steps"])

        utils.write_json({"success_num": success_num, "step_num": step_len, "info": finish_task}, ann_wpath.replace("jsonl", "json"))
        print(f"### finish task num: {len(finish_task.keys())}\tsuccess: {success_num}\tstep_len: {step_len/len(finish_task.keys())}")

    print("output_path: ", ann_wpath)

    print("Creating datasets")
    dataset = create_dataset(config, finish_task)

    print("build android env")

    env = AndroidEnv(config)

    print("Creating agent")
    agent = create_agent(config)

    print("### Start evaluating")

    evaluation(config, agent, dataset, env, ann_wpath)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default="")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)