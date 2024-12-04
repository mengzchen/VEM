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


class DigiRLTrainer():
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
        actor_epochs: int = 3,
    ):
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lm_lr)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.grad_accum_steps = grad_accum_steps
        self.gamma = gamma

        self.actor_epochs = actor_epochs
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

        # TODO next_actions => policy prediction action
        log_prob = self.agent.get_log_prob(history_actions, image_features, next_actions).sum(dim=1).flatten()

        pg_loss = - torch.mean(log_prob * q_values)

        if not validation:
            self.accelerator.backward(pg_loss)

        return pg_loss.detach().cpu().item(), torch.mean(q_values).detach().cpu().item()


    def update_policy(self, replay_buffer, validation_buffer=None):
        self.step += 1
        action_bsize = replay_buffer.batch_size

        logs = []

        print("### updating actor")
        for i in range(self.actor_epochs):
            data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps * action_bsize)]

            for d in data:
                for k, v in d.items():
                    d[k] = v[0]

            dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False)

            dataloader = self.accelerator.prepare(dataloader)
            self.lm_optimizer.zero_grad()

            losses, q_values = [], []
            for batch in dataloader:
                loss, q_value = self.actor_loss(**batch)
                losses.append(loss)
                q_values.append(q_value)

            logging.info(f"epoch: {i}\tloss: {sum(losses) / len(losses):.2f}\tQ-values: {sum(q_values) / len(q_values):.4f}")
            logs.append({"step": self.step * self.actor_epochs + i, "train loss": sum(losses) / len(losses), "train Q value": sum(q_values) / len(q_values)})

            self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.lm_optimizer.step()

        if validation_buffer is not None:
            data = [validation_buffer.sample(1) for _ in range(self.grad_accum_steps * action_bsize)]

            for d in data:
                for k,v in d.items():
                    d[k] = v[0]

            dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False)
            dataloader = self.accelerator.prepare(dataloader)

            with torch.no_grad():
                losses, q_values = [], []
                for batch in dataloader:
                    loss, q_value = self.actor_loss(**batch, validation=True)
                    losses.append(loss)
                    q_values.append(q_value)
                logging.info(f"[val] epoch: {i}\tloss: {sum(losses) / len(losses):.2f}\tQ-values: {sum(q_values) / len(q_values):.4f}")

            logs.append({"step": self.step * self.actor_epochs + i, "val loss": sum(losses) / len(losses), "val Q value": sum(q_values) / len(q_values)})

        return logs


    def infer(self, data, batch_size):
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
            print(outputs)
            for (output, groundtruth, ep_id, step_id) in zip(outputs, groundtruths, ep_ids, step_ids):
                results.append({"output": output, "groundtruth": groundtruth, "ep_id": ep_id, "step_id": step_id.item()})

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
    capacity: int = 500000,
    train_iterations: int = 10,
    epochs:int = 3,
    grad_accum_steps: int = 1,
    lm_lr: float = 1e-5,
    gamma: float = 0.9,
    tau: float = 0.1,
    actor_epochs: int = 3,
    max_grad_norm: float = 0.01,
    save_path: str = None,
    save_freq: int = 25,
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
        actor_epochs=actor_epochs,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm
    )

    # prepare data
    all_trajectories = utils.read_json(data_path)

    agent.prepare()
    trainer.prepare()

    print(f"### all trajectories: {len(all_trajectories)}")

    progress_bar = tqdm(total=train_iterations)
    logs = []
    for i_step in range(train_iterations):
        # sample data for one iteration from buffer, num = actor_epochs * batch_size * grad_accum_steps
        sample_num = actor_epochs * batch_size * grad_accum_steps
        sample_trajectories = random.sample(all_trajectories, sample_num)

        # add predict action to model, multi-GPU gather
        results = trainer.infer(sample_trajectories, batch_size)

        # use results to update sample_trajectories
        assert len(results) == len(sample_trajectories), f"{len(results)} != {sample_num}"
        for i in range(len(results)):
            try:
                actor_output = str_2_format(results[i]["output"])

                if actor_output["action_type"] == "DUAL_POINT":
                    touch_point = actor_output["touch_point"]
                    lift_point = actor_output["lift_point"]
                    click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]

                    click_point = [f"{item:.2f}" for item in click_point]
                    click_point = "({},{})".format(click_point[0], click_point[1])
                    action = f"action_type is click, click_point is {click_point}."
                elif actor_output["action_type"] == "TYPE":
                    action = f"action_type is type, click_point is {actor_output['typed_text']}."
                else:
                    # TODO the press enter is not in critic model
                    action = "action_type is {}.".format(actor_output["action_type"].replace("_", " ").lower())
            except:
                print(F"!!! error parse: {results[i]['output']}")
                action = "action_type is press home"
            sample_trajectories[i]["critic_input"] += action

            sample_trajectories[i]["next_action"] = results[i]["output"]

        # run policy infer to get the predict action
        train_trajectories = sample_trajectories[:int(sample_num * 0.95)]
        val_trajectories = sample_trajectories[int(sample_num * 0.95):]
        replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(train_trajectories))
        validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(val_trajectories))

        for d in train_trajectories:
            replay_buffer.insert(**d)
        for d in val_trajectories:
            validation_buffer.insert(**d)

        print("### training policy")

        logs.extend(trainer.update_policy(replay_buffer, validation_buffer))

        if i_step % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("### saving")
            if not os.path.exists(os.path.join(save_path, f"step_{i_step}")):
                os.mkdir(os.path.join(save_path, f"step_{i_step}"))
            trainer.save(os.path.join(save_path, f"step_{i_step}"))
            utils.write_jsonl(logs, os.path.join(save_path, f"step_{i_step}", "train_log.jsonl"))
            utils.plot_loss(os.path.join(save_path, f"step_{i_step}"), keys=["train loss", "train Q value", "val loss", "val Q value"])

        if accelerator.is_main_process:
            progress_bar.update(1)


def eval_loop(
    agent,
    accelerator,
    eval_path: str = None,
    save_path: str = None,
    batch_size: int = 2,
    **kwargs
):
    result_wpath = os.path.join(save_path, "results.jsonl")
    position_anns = utils.read_json("data/aitw_anns/val_position/aitw_val.json")
    position_dict = {}
    for ann in position_anns:
        position_dict[f"{ann['ep_id']}_{ann['step_id']}"] = ann["annot_position"]

    if os.path.exists(result_wpath):
        print("Baseline AutoUI:")
        baseline_anns = utils.read_jsonl("./checkpoints/Auto-UI-Base/results.jsonl")
        compute_matrix(baseline_anns, position_dict)
        print(f"{' '.join(save_path.split('/')[-2:])}: ")
        anns = utils.read_jsonl(result_wpath)
        compute_matrix(anns, position_dict)
        return

    trainer = DigiRLTrainer(
        agent=agent,
        accelerator=accelerator,
        tokenizer=agent.tokenizer
    )

    # load ckpt
    assert os.path.exists(save_path)
    print(f"### Loading from previous checkpoint: {save_path}")
    trainer.load(save_path)

    trajectories = utils.read_json(eval_path)
    results = trainer.infer(trajectories, batch_size)

    utils.write_jsonl(results, result_wpath)
    compute_matrix(results, position_dict)



@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: "DictConfig"):
    print(OmegaConf.to_yaml(config))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(minutes=40)), kwargs_handlers=[ddp_kwargs], project_dir=config.save_path)
    device = accelerator.device

    print("### load AutoUIAgent")

    agent = AutoUIAgent(device=device, accelerator=accelerator,
                        temperature=config.temperature, do_sample=config.do_sample,
                        policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                        max_new_tokens=config.max_new_tokens)

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