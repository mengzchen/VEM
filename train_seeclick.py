import argparse
import yaml
import os
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
import numpy as np
import ast
import random

import utils
from models.qwen2vl_model import SeeclickAgent
from dataset.digirl_dataset import Qwen2VLDataset, ReplayBuffer, seeclick_action2step
from eval_tools.metrix import str_2_format, check_actions_match
from data_preprocess.prompt import prompt_critic_system, prompt_critic_user
from transformers import PreTrainedModel
from peft import PeftModel



def seeclick_translate_action(step_data):
    try:
        step_data = ast.literal_eval(step_data)
        action_type, touch_point, lift_point, text = step_data["action_type"], [-1.0, -1.0], [-1.0, -1.0], ""

        if action_type == 4:
            action_type = "DUAL_POINT"
            touch_point, lift_point = step_data["click_point"], step_data["click_point"]
        elif action_type == 3:
            action_type = "TYPE"
            text = step_data["typed_text"]
        elif action_type == 0:
            action_type = "SCROLL_DOWN"
            touch_point, lift_point = [0.5, 0.2], [0.5, 0.5]
        elif action_type == 1:
            action_type = "SCROLL_UP"
            touch_point, lift_point = [0.5, 0.5], [0.5, 0.2]
        elif action_type == 8:
            action_type = "SCROLL_LEFT"
            touch_point, lift_point = [0.8, 0.5], [0.2, 0.5]
        elif action_type == 9:
            action_type = "SCROLL_RIGHT"
            touch_point, lift_point = [0.2, 0.5], [0.8, 0.5]        
        elif action_type == 6:
            action_type = "PRESS_HOME"
        elif action_type == 5:
            action_type = "PRESS_BACK"
        elif action_type == 7:
            action_type = "PRESS_ENTER"
        elif action_type == 10:
            action_type = "STATUS_TASK_COMPLETE"
        elif action_type == 11:
            action_type = "STATUS_TASK_IMPOSSIBLE"
        else:
            action_type, touch_point, lift_point, text = "STATUS_TASK_IMPOSSIBLE", [-1.0, -1.0], [-1.0, -1.0], ""
    except:
        print(f"### can't parse the policy output: {step_data}")
        action_type, touch_point, lift_point, text = "STATUS_TASK_IMPOSSIBLE", [-1.0, -1.0], [-1.0, -1.0], ""
    

    action = {"action_type": action_type, "touch_point": touch_point, "lift_point": lift_point, "typed_text": text.lower()}

    return action


def result_dict(step_data):
    try:
        step_data = ast.literal_eval(step_data)
        action_type, touch_point, lift_point, text = step_data["action_type"], [-1.0, -1.0], [-1.0, -1.0], ""

        if action_type == 4:
            action_type_new = "DUAL_POINT"
            touch_point, lift_point = list(step_data["click_point"]), list(step_data["click_point"])
        elif action_type == 3:
            action_type_new = "TYPE"
            text = step_data["typed_text"]
        elif action_type == 0:
            action_type_new = "DUAL_POINT"
            touch_point, lift_point = [0.5, 0.8], [0.5, 0.2]
        elif action_type == 1:
            action_type_new = "DUAL_POINT"
            touch_point, lift_point = [0.5, 0.2], [0.5, 0.8]
        elif action_type == 8:
            action_type_new = "DUAL_POINT"
            touch_point, lift_point = [0.2, 0.5], [0.8, 0.5]
        elif action_type == 9:
            action_type_new = "DUAL_POINT"
            touch_point, lift_point = [0.8, 0.5], [0.2, 0.5]        
        elif action_type == 6:
            action_type_new = "PRESS_HOME"
        elif action_type == 5:
            action_type_new = "PRESS_BACK"
        elif action_type == 7:
            action_type_new = "PRESS_ENTER"
        elif action_type == 10:
            action_type_new = "STATUS_TASK_COMPLETE"
        elif action_type == 11:
            action_type_new = "STATUS_TASK_IMPOSSIBLE"
        else:
            pass
    except:
        action_type, touch_point, lift_point, text = "STATUS_TASK_IMPOSSIBLE", [-1.0, -1.0], [-1.0, -1.0], ""
    

    action = {"action_type": action_type_new, "touch_point": touch_point, "lift_point": lift_point, "typed_text": text.lower()}

    return action


def to_critic(act):
    if act["action_type"] == "DUAL_POINT":
        point = act["touch_point"]
        return f'"action_type": "DUAL_POINT", "click_point": "[{point[0]}, {point[1]}]"'
    elif act["action_type"] == "TYPE":
        text = act['typed_text']
        return f'"action_type": "TYPE", "typed_text": \"{text}\"'
    else:
        action_type = act['action_type']
        return f'"action_type": \"{action_type}\"'
        

def update_trajectory(anns, results):
    for (result, ann) in zip(results, anns):
        new_action = seeclick_translate_action(result["output"])
        new_action_desc = f"step {ann['step_id']}: " + to_critic(new_action)
        
        history_action_desc = "\n".join(ann["action_desc_list"][:ann["step_id"]])
        
        ann["critic_input"] = prompt_critic_system + prompt_critic_user.format(ann["task"], history_action_desc, new_action_desc)
        ann["policy_output"] = seeclick_action2step(new_action)
        ann["critic_image"] = utils.add_visilize2screenshot(ann["policy_image"], new_action, "policy")

    return anns


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
            try:
                pred = result_dict(step["output"])
                groundtruth = str_2_format(step["groundtruth"])
                
                position = position_dict[f"{step['ep_id']}_{step['step_id']}"]
                annot_position = np.array([position[i:i + 4] for i in range(0, len(position), 4)])
                
                check_match = check_actions_match(
                    pred["touch_point"], 
                    pred["lift_point"],
                    pred["action_type"], 
                    groundtruth["touch_point"],
                    groundtruth["lift_point"], 
                    groundtruth["action_type"],
                    annot_position
                )

                if not check_match:
                    # print("pred: ", pred)
                    # print("goal: ", groundtruth)
                    # print("=========")
                    pass
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
        accelerator
    ):
        super().__init__()
        self.agent = agent
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=1e-6)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.grad_accum_steps = config["grad_accum_steps"]
        self.epochs = config["epochs"]
        self.max_grad_norm = 0.01

        self.step = 0
        self.accelerator = accelerator
        self.softmax = torch.nn.Softmax(dim = -1)


    def prepare(self):
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)


    def actor_loss(
        self,
        critic_images,
        critic_inputs,
        policy_inputs,
        policy_outputs,
        policy_images,
        validation=False,
        **kwargs
    ):
        dtype = self.accelerator.unwrap_model(self.agent.model).dtype
        
        messages = []
        for critic_input, critic_image in zip(critic_inputs, critic_images):
            try:
                messages.append([{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": critic_input},
                        {"type": "image", "image": critic_image, "max_pixels": 56000}
                    ]
                }])
            except:
                print(f"!!! error image: {critic_image}")

        texts = self.agent.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        vision_inputs = []
        for message in messages:
            vision_input, _ = process_vision_info(message)
            vision_inputs.append(vision_input)

        inputs = self.agent.processor(
            text=texts,
            images=vision_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.agent.critic.device)
        
        generated_ids = self.agent.critic.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        q_values = self.agent.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        q_values = [int(val) for val in q_values]
        q_values = torch.tensor(q_values, dtype=dtype, requires_grad=True).to(self.agent.model.device)

        q_values = q_values / 2

        log_prob = self.agent.get_log_prob(policy_inputs, policy_images, policy_outputs).sum(dim=1).flatten()
        pg_loss = - torch.mean(log_prob * q_values)

        if not validation:
            self.accelerator.backward(pg_loss)

        return pg_loss.detach().cpu().item(), torch.mean(q_values).detach().cpu().item()


    def update_policy(self, buffer, is_validation, batch_size):
        logs = []

        self.step += 1
        data = [buffer.sample(1) for _ in range(self.grad_accum_steps * batch_size)]

        for d in data:
            for k, v in d.items():
                d[k] = v[0]

        dataloader = self.accelerator.prepare(DataLoader(Qwen2VLDataset(data, is_train=True), batch_size=batch_size, shuffle=False))

        self.lm_optimizer.zero_grad()
        losses, q_values = [], []
        if is_validation:
            with torch.no_grad():
                for batch in dataloader:
                    loss, q_value = self.actor_loss(**batch, validation=True)
                    losses.append(loss)
                    q_values.append(q_value)
            logging.info(f"[val] step: {self.step}\tloss: {sum(losses) / len(losses):.2f}\tQ-values: {sum(q_values) / len(q_values):.4f}")
            logs.append({"step": self.step, "val loss": sum(losses) / len(losses), "val Q value": sum(q_values) / len(q_values)})
        else:
            for batch in dataloader:
                loss, q_value = self.actor_loss(**batch)
                losses.append(loss)
                q_values.append(q_value)

            logging.info(f"step: {self.step}\tloss: {sum(losses) / len(losses):.2f}\tQ-values: {sum(q_values) / len(q_values):.4f}")
            logs.append({"step": self.step, "train loss": sum(losses) / len(losses), "train Q value": sum(q_values) / len(q_values)})

            self.accelerator.clip_grad_norm_(self.agent.parameters(), 0.01)
            self.lm_optimizer.step()

        return logs


    def infer(self, anns, batch_size):
        dataloader = DataLoader(Qwen2VLDataset(anns, is_train=False), batch_size=batch_size, shuffle=False)
        dataloader = self.accelerator.prepare(dataloader)

        results = []
        for batch in dataloader:
            ep_ids, step_ids = batch["ep_id"], batch["step_id"]
            texts, groundtruths, image_paths = batch["policy_input"], batch["policy_output"], batch["policy_image"]

            outputs = self.agent.get_action(texts, image_paths)

            for (output, groundtruth, ep_id, step_id) in zip(outputs, groundtruths, ep_ids, step_ids):
                results.append({"output": output, "groundtruth": groundtruth, "ep_id": ep_id, "step_id": step_id.item()})

        return results


    def save(self, path):
        self.accelerator.save_state(path, safe_serialization=False)
    

    def _save(self, output_dir, state_dict):
        os.makedirs(output_dir, exist_ok=True)

        self.agent.model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=True
        )


    def load(self, path):
        self.accelerator.load_state(path)


def get_peft_state_maybe_zero_3(named_params):
    to_return = {k: t for k, t in named_params if "lora_" in k}
    
    to_return = {k: v.detach().cpu().clone() for k, v in to_return.items()}

    return to_return



def train(agent, accelerator, config):
    batch_size = config["batch_size"]
    trainer = DigiRLTrainer(
        config=config,
        agent=agent,
        accelerator=accelerator
    )

    all_trajectories = utils.read_jsonl(config["train_data"])

    print(f"### all trajectories: {len(all_trajectories)}")

    logs = []
    train_trajectories = all_trajectories[:int(len(all_trajectories) * 0.95)]
    val_trajectories = all_trajectories[int(len(all_trajectories) * 0.95):]
    random.shuffle(train_trajectories)
    sample_num = batch_size * config["grad_accum_steps"]

    for epoch in range(config["epochs"]):
        print(f"### epoch {epoch}")
        for train_step in range(len(train_trajectories) // sample_num):
            sample_trajectories = train_trajectories[train_step * sample_num: (train_step + 1) * sample_num]

            results = trainer.infer(sample_trajectories, batch_size)
            sample_trajectories = update_trajectory(sample_trajectories, results)
            
            replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(sample_trajectories))

            for d in sample_trajectories:
                replay_buffer.insert(**d)

            logs.extend(trainer.update_policy(replay_buffer, is_validation=False, batch_size=batch_size))

        if accelerator.is_main_process:
            save_path = f"checkpoints/{config['model_name']}"
            print(f"### save model at: {save_path}")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(os.path.join(save_path, f"epoch_{epoch}")):
                os.mkdir(os.path.join(save_path, f"epoch_{epoch}"))

            state_dict = get_peft_state_maybe_zero_3(trainer.agent.model.named_parameters())
            trainer._save(os.path.join(save_path, f"epoch_{epoch}"), state_dict=state_dict)

            utils.write_jsonl(logs, os.path.join(save_path, f"epoch_{epoch}", "train_log.jsonl"))
            utils.plot_loss(os.path.join(save_path, f"epoch_{epoch}"), keys=["train loss", "train Q value", "val loss", "val Q value"])
        
        results = trainer.infer(val_trajectories, batch_size)
        val_trajectories = update_trajectory(val_trajectories, results)
        validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(val_trajectories))
        for d in val_trajectories:
            validation_buffer.insert(**d)
        logs.extend(trainer.update_policy(validation_buffer, is_validation=True, batch_size=batch_size))
        


def evaluation(agent, accelerator, config):
    result_dir = f"checkpoints/{config['test_task']}_result"
    result_wpath = os.path.join(result_dir, f"{config['test_task']}_{config['model_name']}_results.jsonl")

    print(f"### result path: {result_wpath}")
    
    anns = utils.read_jsonl(config['eval_data'])
    
    position_dict = {}
    for ann in anns:
        position_dict[f"{ann['ep_id']}_{ann['step_id']}"] = ann["position"]

    if not os.path.exists(result_wpath):
        trainer = DigiRLTrainer(
            config=config,
            agent=agent,
            accelerator=accelerator
        )

        results = trainer.infer(anns, config["batch_size"])
        utils.write_jsonl(results, result_wpath)
    else:
        results = utils.read_jsonl(result_wpath)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for file in os.listdir(result_dir):
        result_wpath = os.path.join(result_dir, file)
        results = utils.read_jsonl(result_wpath)
        print(f"================{result_wpath.split('/')[2]}================")
        compute_matrix(results, position_dict)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--eval', action='store_true')  
    args = parser.parse_args()

    if args.task == "webshop":
        config = "configs/policy/rl_seeclick_webshop.yaml"
    else:
        config = "configs/policy/rl_seeclick_general.yaml"

    with open(config, 'r') as file:
        config = yaml.safe_load(file)
        
    print(f"### config:\n{config}")
    
    accelerator = Accelerator()

    print("### load SeeclickAgent")

    agent = SeeclickAgent(accelerator=accelerator, config=config, is_eval=args.eval)
    
    if args.eval:
        evaluation(agent=agent, accelerator=accelerator, config=config)
    else:
        train(agent=agent, accelerator=accelerator, config=config)
