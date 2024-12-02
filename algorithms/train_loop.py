from datasets.digirl_dataset import ReplayBuffer
import numpy as np
from tqdm import tqdm
from algorithms.digirl import DigiRLTrainer
import os
import torch
import time
import copy
import utils
import accelerate
import utils


def label_trajectories(trajectories, agent):
    print("Labeling Trajectories")
    baselines = []
    for i in range(0, len(trajectories), 16):
        observations = [t[0]["observation"] for t in trajectories[i:i+16]]
        with torch.no_grad():
            v = agent.trajectory_critic(observations)
            v = torch.nn.Softmax(dim = -1)(v)[:,1]
            baselines.append(v.flatten())
    baselines = torch.cat(baselines, dim = -1)
    print("Done Labeling Trajectories")
    return torch.clamp(baselines.cpu(), 1e-4, 1-1e-4)


def onpolicy_train_loop(
    agent,
    tokenizer,
    accelerator,
    data_path: str = None, 
    warmup_iter: int = 20,
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
        tokenizer=tokenizer,
        lm_lr=lm_lr,
        gamma=gamma,
        tau=tau,
        epochs=epochs,
        actor_epochs=actor_epochs,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm
    )
    
    # prepare data
    replay_buffer= ReplayBuffer(batch_size=batch_size, capacity=capacity)
    all_trajectories = torch.load(data_path) # TODO use ours data
    
    agent.prepare()
    trainer.prepare()
            
    train_trajectories = all_trajectories[:int(len(all_trajectories)*0.95)]
    val_trajectories = all_trajectories[int(len(all_trajectories)*0.95):]
    utils.colorful_print(f"### train: {len(train_trajectories)} val: {len(val_trajectories)}", "green")

    replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)

    data = sum(train_trajectories, [])
    val_data = sum(val_trajectories, [])
    for d in data:
        replay_buffer.insert(**d)
    for d in val_data:
        validation_buffer.insert(**d)
    
    progress_bar = tqdm(total=train_iterations)
    
    for i in range(train_iterations):
        # TODO make sure the impact of deleting code of generating more data 
        
        utils.colorful_print("### training policy", "green")
        
        trainer.update(replay_buffer, validation_buffer, no_update_actor=(i < warmup_iter))
    
        if (i+1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            utils.colorful_print("### saving", "green")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            
        if accelerator.is_main_process:
            progress_bar.update(1)
        
