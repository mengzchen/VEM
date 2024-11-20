from models.autoui_agent import AutoUIAgent
from algorithms.train_loop import onpolicy_train_loop
from algorithms.eval_loop import eval_loop 
from omegaconf import DictConfig, OmegaConf
import hydra
import os
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
import utils


def load_task_file(assets_path, task_set, task_split):
    all_tasks = []
    with open(os.path.join(assets_path, task_set + "_" + task_split + ".txt")) as fb: 
        for line in fb:
            all_tasks.append(line.replace("\n", ""))
    return all_tasks


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: "DictConfig"):
    utils.colorful_print(OmegaConf.to_yaml(config), "blue")
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(minutes=40)), kwargs_handlers=[ddp_kwargs], project_dir=config.save_path)
    device = accelerator.device

    utils.colorful_print("### load AutoUIAgent", "green")

    agent = AutoUIAgent(device=device, accelerator=accelerator, 
                        temperature=config.temperature, do_sample=config.do_sample, 
                        policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                        max_new_tokens=config.max_new_tokens)
    tokenizer = agent.tokenizer

    onpolicy_train_loop(
        tokenizer=tokenizer,
        agent=agent,
        accelerator=accelerator,
        **config
    )
        
    eval_loop(
        tokenizer=tokenizer,
        agent = agent,
        accelerator = accelerator,
        decode_f=lambda x:x,
        **config
    )
    

if __name__ == "__main__":
    main()
