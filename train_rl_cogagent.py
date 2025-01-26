import argparse
import yaml
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from models.cogagent_model import CogAgent
from dataset.digirl_dataset import CogAgentDataset
from eval_tools.metrix import str_2_format, check_actions_match
import numpy as np
from data_preprocess.utils import cogagent_translate_action, ActionType


def convert_format(response):
    action_class = cogagent_translate_action(response)
    if action_class.action_type in [ActionType.DualPoint, ActionType.Up, ActionType.Down, ActionType.Left, ActionType.Right]:
        return {"action_type": "DUAL_POINT", "touch_point": action_class.touch_point, "lift_point": action_class.lift_point, "typed_text": ""}
    elif action_class.action_type == ActionType.Type:
        return {"action_type": "TYPE", "touch_point": [-1.0, -1.0], "lift_point": [-1.0, -1.0], "typed_text": action_class.typed_text}
    elif action_class.action_type == ActionType.GoBack:
        return {"action_type": "PRESS_BACK", "touch_point": [-1.0, -1.0], "lift_point": [-1.0, -1.0], "typed_text": ""}
    elif action_class.action_type == ActionType.GoHome:
        return {"action_type": "PRESS_HOME", "touch_point": [-1.0, -1.0], "lift_point": [-1.0, -1.0], "typed_text": ""}
    elif action_class.action_type == ActionType.Enter:
        return {"action_type": "PRESS_ENTER", "touch_point": [-1.0, -1.0], "lift_point": [-1.0, -1.0], "typed_text": ""}
    elif action_class.action_type == ActionType.TaskComplete or action_class.action_type == ActionType.TaskImpossible:
        return {"action_type": "STATUS_TASK_COMPLETE", "touch_point": [-1.0, -1.0], "lift_point": [-1.0, -1.0], "typed_text": ""}
    else:
        print(f"Action {action_class} not supported yet.")
        return ""


def compute_matrix(anns, position_dict):
    ep2ann = {}
    for ann in anns:
        if ann["ep_id"] not in ep2ann.keys():
            ep2ann[ann["ep_id"]] = []
        ep2ann[ann["ep_id"]].append(ann)

    succ_task, task_num = 0, 0
    succ_step, step_num = 0, 0
    for _, ann in ep2ann.items():
        task_flag = True
        task_num += 1
        for step in ann:
            step_num += 1
            
            pred = convert_format(step["output"])
            groundtruth = str_2_format(step["groundtruth"])
            print(step["output"])
            print(pred)
            print(step["groundtruth"])
            print(groundtruth)
            print("===========")
            
            position = position_dict[f"{step['ep_id']}_{step['step_id']}"]
            annot_position = np.array([position[i:i + 4] for i in range(0, len(position), 4)])
            
            try:
                check_match = check_actions_match(
                    pred["touch_point"], 
                    pred["lift_point"],
                    pred["action_type"], 
                    groundtruth["touch_point"],
                    groundtruth["lift_point"], 
                    groundtruth["action_type"],
                    annot_position
                )
            except:
                print("error")
                check_match = False
            
            if check_match == True:
                succ_step += 1
            else:
               task_flag = False

        if task_flag: succ_task += 1

    step_succ_rate = succ_step / step_num
    task_succ_rate = succ_task / task_num

    print(f"step succ rate: {str(step_succ_rate)} ({succ_step}/{step_num})")
    print(f"task succ rate: {str(task_succ_rate)} ({succ_task}/{task_num})")


class DigiRLTrainer:
    def __init__(self,
        config, 
        agent,
        accelerator,
        tokenizer
    ):
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.accelerator = accelerator


    def prepare(self):
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)


    def infer(self, anns, batch_size):
        dataloader = DataLoader(CogAgentDataset(anns), batch_size=batch_size, shuffle=False)
        dataloader = self.accelerator.prepare(dataloader)

        results = []
        for batch in tqdm(dataloader):
            ep_ids, step_ids = batch["ep_id"], batch["step_id"]
            texts, groundtruths, image_paths = batch["policy_input"], batch["policy_output"], batch["policy_image"]

            outputs = self.agent.get_action(texts, image_paths)

            for (output, groundtruth, ep_id, step_id) in zip(outputs, groundtruths, ep_ids, step_ids):
                results.append({"output": output, "groundtruth": groundtruth, "ep_id": ep_id, "step_id": step_id.item()})

        return results


    def load(self, path):
        self.accelerator.load_state(path)


def evaluation(agent, accelerator, config):
    result_dir = f"checkpoints/{config['test_task']}_result"
    result_wpath = os.path.join(result_dir, f"{config['test_task']}_{config['model_name']}_results.jsonl")
    save_model = f"checkpoints/{config['model_name']}"
    print(f"### result path: {result_wpath}")
    
    anns = utils.read_jsonl(config['eval_data'])
    
    position_dict = {}
    for ann in anns:
        position_dict[f"{ann['ep_id']}_{ann['step_id']}"] = ann["position"]

    if not os.path.exists(result_wpath):
        trainer = DigiRLTrainer(
            config=config,
            agent=agent,
            accelerator=accelerator,
            tokenizer=agent.tokenizer
        )

        assert os.path.exists(save_model)
        print(f"### Loading from previous checkpoint: {save_model}")
        trainer.load(save_model)

        results = trainer.infer(anns, config["batch_size"])
        utils.write_jsonl(results, result_wpath)
    else:
        results = utils.read_jsonl(result_wpath)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    compute_matrix(results, position_dict)



def main(config):
    accelerator = Accelerator()

    print("### load CogAgent")

    agent = CogAgent(accelerator=accelerator, config=config)

    evaluation(agent=agent, accelerator=accelerator, config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    config = "configs/policy/rl_cogagent.yaml"
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
