from omegaconf import DictConfig, OmegaConf
import hydra
import os
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
import random

import utils
from dataset.digirl_dataset import ReplayBuffer 
from VLAM.models.cogagent_model import Qwen2VLAgent
from eval_tools.metrix import compute_matrix
from data_preprocess.utils import update_trajectory
from train_rl import DigiRLTrainer



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

    all_trajectories = utils.read_jsonl(data_path)

    agent.prepare()
    trainer.prepare()

    print(f"### all trajectories: {len(all_trajectories)}")

    logs = []
    train_trajectories = all_trajectories[:int(len(all_trajectories) * 0.95)]
    val_trajectories = all_trajectories[int(len(all_trajectories) * 0.95):]
    random.shuffle(train_trajectories)
    sample_num = batch_size * grad_accum_steps
    for epoch in range(epochs):
        print(f"### epoch {epoch}")
        for train_step in range(len(train_trajectories) // sample_num):
            sample_trajectories = train_trajectories[train_step * sample_num: (train_step + 1) * sample_num]

            results = trainer.infer(sample_trajectories, batch_size)
            sample_trajectories = update_trajectory(sample_trajectories, results)
            
            replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(sample_trajectories))

            for d in sample_trajectories:
                replay_buffer.insert(**d)

            logs.extend(trainer.update_policy(replay_buffer, is_validation=False, batch_size=batch_size))

        results = trainer.infer(val_trajectories, batch_size)
        val_trajectories = update_trajectory(val_trajectories, results)
        validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(val_trajectories))
        for d in val_trajectories:
            validation_buffer.insert(**d)
        logs.extend(trainer.update_policy(validation_buffer, is_validation=True, batch_size=batch_size))

        if accelerator.is_main_process:
            print("### saving")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(os.path.join(save_path, f"epoch_{epoch}")):
                os.mkdir(os.path.join(save_path, f"epoch_{epoch}"))
            trainer.save(os.path.join(save_path, f"epoch_{epoch}"))
            utils.write_jsonl(logs, os.path.join(save_path, f"epoch_{epoch}", "train_log.jsonl"))
            utils.plot_loss(os.path.join(save_path, f"epoch_{epoch}"), keys=["train loss", "train Q value", "val loss", "val Q value"])


def eval_loop(
    agent,
    accelerator,
    eval_path,
    save_path,
    batch_size,
    test_task,
    **kwargs
):
    model_name = "_".join(save_path.split("/")[-2:]).replace("checkpoints_", "")
    result_dir = f"checkpoints/{test_task}_result"
    result_wpath = os.path.join(result_dir, f"{test_task}_{model_name}_results.jsonl")
    print(f"### result path: {result_wpath}")
    
    position_anns = utils.read_jsonl(eval_path)
    
    position_dict = {}
    for ann in position_anns:
        position_dict[f"{ann['ep_id']}_{ann['step_id']}"] = ann["position"]

    if not os.path.exists(result_wpath):
        trainer = DigiRLTrainer(
            agent=agent,
            accelerator=accelerator,
            tokenizer=agent.tokenizer
        )

        assert os.path.exists(save_path)
        print(f"### Loading from previous checkpoint: {save_path}")
        # trainer.load(save_path)

        trajectories = utils.read_jsonl(eval_path)
        results = trainer.infer(trajectories, batch_size)
        utils.write_jsonl(results, result_wpath)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for file in os.listdir(result_dir):
        result_wpath = os.path.join(result_dir, file)
        results = utils.read_jsonl(result_wpath)
        print(f"================{result_wpath.split('/')[2]}================")
        compute_matrix(results, position_dict)



@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: "DictConfig"):
    print(OmegaConf.to_yaml(config))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        InitProcessGroupKwargs(timeout=timedelta(minutes=40)),
        kwargs_handlers=[ddp_kwargs],
        project_dir=config.save_path
    )

    print("### load Qwen2VLAgent")

    agent = Qwen2VLAgent(
        accelerator=accelerator,
        policy_lm=config.policy_lm,
        critic_lm=config.critic_lm
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

