from models.autoui_agent import AutoUIAgent
from omegaconf import DictConfig, OmegaConf
import hydra
import os
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
import utils

from datasets.digirl_dataset import ReplayBuffer, DummyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
from qwen_vl_utils import process_vision_info
from models.autoui_agent import ImageFeatureExtractor

from eval_tools.aitw import compute_matrix
import logging
import random
from eval_tools.aitw import str_2_format
from data_preprocess.action_transfer import update_trajectory


class DigiRLTrainer:
    def __init__(self,
        agent,
        accelerator,
        tokenizer,
        lm_lr: float = 1e-5,
        grad_accum_steps: int = 8,
        gamma: float = 0.9,
        tau: float = 0.1,
        epochs: int = 3,
        max_grad_norm: float=0.01,
    ):
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lm_lr)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.grad_accum_steps = grad_accum_steps
        self.gamma = gamma

        self.epochs = epochs

        self.step = 0
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.softmax = torch.nn.Softmax(dim = -1)
        self.image_process = ImageFeatureExtractor()


    def prepare(self):
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)


    def actor_loss(
        self,
        image_paths,
        critic_inputs,
        history_actions,
        next_actions,
        validation=False,
        **kwargs
    ):
        dtype = self.accelerator.unwrap_model(self.agent.model).dtype
        device = self.accelerator.unwrap_model(self.agent.model).device

        messages = []
        for critic_input, image_path in zip(critic_inputs, image_paths):
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "text", "text": critic_input},
                    {"type": "image", "image": image_path, "max_pixels": 56000}
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

        q_values = (q_values - 2) / 2

        image_features = []
        for image_path in image_paths:
            image_feature = self.image_process.to_feat(image_path=image_path).to(device, dtype=dtype)
            image_features.append(image_feature)
        image_features = torch.stack(image_features)

        log_prob = self.agent.get_log_prob(history_actions, image_features, next_actions).sum(dim=1).flatten()

        pg_loss = - torch.mean(log_prob * q_values)

        if not validation:
            self.accelerator.backward(pg_loss)

        return pg_loss.detach().cpu().item(), torch.mean(q_values).detach().cpu().item()


    def update_policy(self, buffer, is_validation, batch_size):
        logs = []

        self.step += 1
        data = [buffer.sample(1) for _ in range(self.grad_accum_steps * batch_size)]

        # TODO no need for buffer
        for d in data:
            for k, v in d.items():
                d[k] = v[0]

        dataloader = self.accelerator.prepare(DataLoader(DummyDataset(data), batch_size=batch_size, shuffle=False))

        self.lm_optimizer.zero_grad()

        losses, q_values = [], []
        if is_validation:
            with torch.no_grad():
                for batch in dataloader:
                    loss, q_value = self.actor_loss(**batch)
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

            self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.lm_optimizer.step()

        return logs


    def infer(self, data, batch_size, add_q_value):
        dtype = self.accelerator.unwrap_model(self.agent.model).dtype
        device = self.accelerator.unwrap_model(self.agent.model).device
        dataloader = DataLoader(DummyDataset(data), batch_size=batch_size, shuffle=False)
        dataloader = self.accelerator.prepare(dataloader)

        results = []
        for batch in tqdm(dataloader):
            ep_ids, step_ids = batch["ep_id"], batch["step_id"]
            texts, groundtruths = batch["history_action"], batch["next_action"]
            image_paths = batch["image_path"]
            image_features = []
            for image_path in image_paths:
                image_feature = self.image_process.to_feat(image_path=image_path).to(device, dtype=dtype)
                image_features.append(image_feature)
            image_features = torch.stack(image_features)

            outputs = self.agent.get_action(texts, image_features)

            for (output, groundtruth, ep_id, step_id) in zip(outputs, groundtruths, ep_ids, step_ids):
                results.append({"output": output, "groundtruth": groundtruth, "ep_id": ep_id, "step_id": step_id.item()})

        if add_q_value:
            q_values = []
            data = update_trajectory(data, results)
            buffer = ReplayBuffer(batch_size=batch_size, capacity=len(data))

            for d in data:
                buffer.insert(**d)
            data = [buffer.sample(1) for _ in range(len(data))]

            for d in data:
                for k, v in d.items():
                    d[k] = v[0]

            dataloader = DataLoader(DummyDataset(data), batch_size=batch_size, shuffle=False)
            dataloader = self.accelerator.prepare(dataloader)
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    loss, q_value = self.actor_loss(**batch, validation=True)
                    q_values.append(q_value)

            return results, sum(q_values) / len(q_values)

        return results


    def save(self, path):
        self.accelerator.save_state(path, safe_serialization=False)


    def load(self, path):
        self.accelerator.load_state(path)


def onpolicy_train_loop(
    agent,
    accelerator,
    data_path: str = None,
    batch_size: int = 2,
    epochs: int = 3,
    grad_accum_steps: int = 1,
    lm_lr: float = 1e-5,
    gamma: float = 0.9,
    tau: float = 0.1,
    max_grad_norm: float = 0.01,
    save_path: str = None,
    **kwargs
):
    trainer = DigiRLTrainer(
        agent=agent,
        accelerator=accelerator,
        tokenizer=agent.tokenizer,
        lm_lr=lm_lr,
        gamma=gamma,
        tau=tau,
        epochs=epochs,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm
    )

    all_trajectories = utils.read_json(data_path)

    agent.prepare()
    trainer.prepare()

    print(f"### all trajectories: {len(all_trajectories)}")

    logs = []
    # split val and train
    train_trajectories = all_trajectories[:int(len(all_trajectories) * 0.95)]
    val_trajectories = all_trajectories[int(len(all_trajectories) * 0.95):]
    random.shuffle(train_trajectories)
    sample_num = batch_size * grad_accum_steps
    for epoch in range(epochs):
        print(f"### epoch {epoch}")
        for train_step in range(len(train_trajectories) // sample_num):
            sample_trajectories = train_trajectories[train_step * sample_num: (train_step + 1) * sample_num]

            results = trainer.infer(sample_trajectories, batch_size, add_q_value=False)
            sample_trajectories = update_trajectory(sample_trajectories, results)
            replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(sample_trajectories))

            for d in sample_trajectories:
                replay_buffer.insert(**d)

            logs.extend(trainer.update_policy(replay_buffer, is_validation=False, batch_size=batch_size))

        validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(val_trajectories))
        for d in val_trajectories:
            validation_buffer.insert(**d)
        logs.extend(trainer.update_policy(validation_buffer, is_validation=True, batch_size=batch_size))

        if accelerator.is_main_process:
            print("### saving")
            if not os.path.exists(os.path.join(save_path, f"epoch_{epoch}")):
                os.mkdir(os.path.join(save_path, f"epoch_{epoch}"))
            trainer.save(os.path.join(save_path, f"epoch_{epoch}"))
            utils.write_jsonl(logs, os.path.join(save_path, f"epoch_{epoch}", "train_log.jsonl"))
            utils.plot_loss(os.path.join(save_path, f"epoch_{epoch}"), keys=["train loss", "train Q value", "val loss", "val Q value"])


def eval_loop(
    agent,
    accelerator,
    eval_path: str = None,
    save_path: str = None,
    batch_size: int = 2,
    **kwargs
):
    model_name = "_".join(save_path.split("/")[-2:]).replace("checkpoints_", "")
    # TODO change the name of result_wpath
    result_wpath = os.path.join("checkpoints/results", f"{model_name}_val_results.jsonl")

    position_anns = utils.read_json("data/aitw_anns/aitw_position_val.json")
    position_dict = {}
    for ann in position_anns:
        position_dict[f"{ann['ep_id']}_{ann['step_id']}"] = ann["annot_position"]

    if not os.path.exists(result_wpath):
        trainer = DigiRLTrainer(
            agent=agent,
            accelerator=accelerator,
            tokenizer=agent.tokenizer
        )

        assert os.path.exists(save_path)
        print(f"### Loading from previous checkpoint: {save_path}")
        trainer.load(save_path)

        trajectories = utils.read_json(eval_path)
        _, q_values = trainer.infer(trajectories, batch_size, add_q_value=True)
        print(f"### {model_name} q_values: {q_values}")

    # for file in os.listdir("checkpoints/results"):
    #     result_wpath = os.path.join("checkpoints/results", file)
    #     results = utils.read_jsonl(result_wpath)
    #     print(f"================{result_wpath.split('/')[2]}================")
    #     compute_matrix(results, position_dict)



@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: "DictConfig"):
    print(OmegaConf.to_yaml(config))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        InitProcessGroupKwargs(timeout=timedelta(minutes=40)),
        kwargs_handlers=[ddp_kwargs],
        project_dir=config.save_path
    )
    device = accelerator.device

    print("### load AutoUIAgent")

    agent = AutoUIAgent(
        device=device,
        accelerator=accelerator,
        temperature=config.temperature,
        do_sample=config.do_sample,
        policy_lm=config.policy_lm,
        critic_lm=config.critic_lm,
        max_new_tokens=config.max_new_tokens
    )

    if config.eval_only:
        eval_loop(
            agent=agent,
            accelerator=accelerator,
            **config
        )
    else:
        onpolicy_train_loop(
            agent=agent,
            accelerator=accelerator,
            **config
        )


if __name__ == "__main__":
    main()