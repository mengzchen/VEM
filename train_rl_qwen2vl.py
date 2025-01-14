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

import utils
from models.qwen2vl_model import Qwen2VLAgent
from dataset.digirl_dataset import Qwen2VLDataset
from eval_tools.metrix import str_2_format, check_actions_match


def qwen2vl_translate_action(step_data):
    step_data = ast.literal_eval(step_data)
    action_type, touch_point, lift_point, text = step_data["action_type"], [-1.0, -1.0], [-1.0, -1.0], ""

    if action_type == 4:
        action_type_new = "DUAL_POINT"
        touch_point, lift_point = step_data["click_point"], step_data["click_point"]
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
    

    action = {"action_type": action_type_new, "touch_point": touch_point, "lift_point": lift_point, "typed_text": text.lower()}

    action["touch_point"] = [action["touch_point"][1], action["touch_point"][0]]
    action["lift_point"] = [action["lift_point"][1], action["lift_point"][0]]

    return action


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
                pred = qwen2vl_translate_action(step["output"])
                groundtruth = str_2_format(step["groundtruth"])
                
                print(pred)
                print(groundtruth)
                print("=========")
                
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
        device = self.accelerator.unwrap_model(self.agent.model).device
        
        messages = []
        for critic_input, critic_image in zip(critic_inputs, critic_images):
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "text", "text": critic_input},
                    {"type": "image", "image": critic_image, "max_pixels": 56000}
                ]
            }])

        texts = self.agent.critic_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        vision_inputs = []
        for message in messages:
            vision_input, _ = process_vision_info(message)
            vision_inputs.append(vision_input)

        inputs = self.agent.critic_processor(
            text=texts,
            images=vision_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        generated_ids = self.agent.critic.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        q_values = self.agent.critic_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        q_values = [int(val) for val in q_values]
        q_values = torch.tensor(q_values, dtype=dtype, requires_grad=True).to(device)

        q_values = q_values / 2

        policy_image_features = []
        for policy_image in policy_images:
            policy_image_feature = self.image_process.to_feat(image_path=policy_image).to(device, dtype=dtype)
            policy_image_features.append(policy_image_feature)
        policy_image_features = torch.stack(policy_image_features)

        log_prob = self.agent.get_log_prob(policy_inputs, policy_image_features, policy_outputs).sum(dim=1).flatten()

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

        keys = ["ep_id", "step_id", "policy_inputs", "policy_outputs", "policy_images", "critic_inputs", "critic_images"]
        dataloader = self.accelerator.prepare(DataLoader(DummyDataset(data, keys), batch_size=batch_size, shuffle=False))

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
        dataloader = DataLoader(Qwen2VLDataset(anns), batch_size=batch_size, shuffle=False)
        dataloader = self.accelerator.prepare(dataloader)

        results = []
        for batch in tqdm(dataloader):
            ep_ids, step_ids = batch["ep_id"], batch["step_id"]
            texts, groundtruths, image_paths = batch["policy_input"], batch["policy_output"], batch["policy_image"]

            outputs = self.agent.get_action(texts, image_paths)

            for (output, groundtruth, ep_id, step_id) in zip(outputs, groundtruths, ep_ids, step_ids):
                results.append({"output": output, "groundtruth": groundtruth, "ep_id": ep_id, "step_id": step_id.item()})

        return results


    def save(self, path):
        self.accelerator.save_state(path, safe_serialization=False)


    def load(self, path):
        self.accelerator.load_state(path)


# def train(agent, accelerator, config):
#     trainer = DigiRLTrainer(
#         agent=agent,
#         accelerator=accelerator,
#         tokenizer=agent.policy_tokenizer,
#         lm_lr=config["lm_lr"],
#         gamma=config["gamma"],
#         tau=config["tau"],
#         epochs=config["epochs"],
#         grad_accum_steps=config["grad_accum_steps"],
#         max_grad_norm=config["max_grad_norm"]
#     )

#     all_trajectories = utils.read_jsonl(config["train_data"])

#     agent.prepare()
#     trainer.prepare()

#     print(f"### all trajectories: {len(all_trajectories)}")

#     logs = []
#     train_trajectories = all_trajectories[:int(len(all_trajectories) * 0.95)]
#     val_trajectories = all_trajectories[int(len(all_trajectories) * 0.95):]
#     random.shuffle(train_trajectories)
#     sample_num = batch_size * grad_accum_steps
#     for epoch in range(epochs):
#         print(f"### epoch {epoch}")
#         for train_step in range(len(train_trajectories) // sample_num):
#             sample_trajectories = train_trajectories[train_step * sample_num: (train_step + 1) * sample_num]

#             results = trainer.infer(sample_trajectories, batch_size)
#             sample_trajectories = update_trajectory(sample_trajectories, results)
            
#             replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(sample_trajectories))

#             for d in sample_trajectories:
#                 replay_buffer.insert(**d)

#             logs.extend(trainer.update_policy(replay_buffer, is_validation=False, batch_size=batch_size))

#         results = trainer.infer(val_trajectories, batch_size)
#         val_trajectories = update_trajectory(val_trajectories, results)
#         validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(val_trajectories))
#         for d in val_trajectories:
#             validation_buffer.insert(**d)
#         logs.extend(trainer.update_policy(validation_buffer, is_validation=True, batch_size=batch_size))

#         if accelerator.is_main_process:
#             print("### saving")
#             if not os.path.exists(save_path):
#                 os.mkdir(save_path)
#             if not os.path.exists(os.path.join(save_path, f"epoch_{epoch}")):
#                 os.mkdir(os.path.join(save_path, f"epoch_{epoch}"))
#             trainer.save(os.path.join(save_path, f"epoch_{epoch}"))
#             utils.write_jsonl(logs, os.path.join(save_path, f"epoch_{epoch}", "train_log.jsonl"))
#             utils.plot_loss(os.path.join(save_path, f"epoch_{epoch}"), keys=["train loss", "train Q value", "val loss", "val Q value"])


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
            accelerator=accelerator
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

    print("### load Qwen2VLAgent")

    agent = Qwen2VLAgent(accelerator=accelerator, config=config)

    if config["eval_only"]:
        evaluation(agent=agent, accelerator=accelerator, config=config)
    else:
        train(agent=agent, accelerator=accelerator, config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    config = "configs/policy/rl_qwen2vl.yaml"
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
