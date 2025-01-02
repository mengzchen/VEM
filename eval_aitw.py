import argparse
import yaml
import json
import os
from accelerate import DistributedDataParallelKwargs, Accelerator

import utils
from dataset import create_dataset
from models import create_model
from eval_tools.androidenv import AndroidEnv




def evaluation(config, model, dataloader, env):
    all_trajectories = []
    for i in range(eval_iterations):
        trajectories = batch_interact_environment(
            agent = agent,
            env = env,
            num_trajectories= rollout_size,\
            accelerator = accelerator,\
            gamma = gamma,
            iter=i
        )
        if accelerator.is_main_process:
            info = {"iteration": i,\
                    "rollout.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "rollout.max": np.max([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "rollout.min": np.min([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "walltime": time.time()}
            all_trajectories += trajectories
            
            torch.save(all_trajectories, os.path.join(save_path, 'trajectories_eval.pt'))
            time.sleep(15)
        else:
            info = {}
        accelerator.wait_for_everyone()
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories_eval.pt'))
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)


def main(config):
    print("config:", json.dumps(config))
    output_path = os.path.join("checkpoints/results/", f"{config['model_name']}.jsonl")
    print("output_path: ", output_path)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    print("### build android env")

    env = AndroidEnv(config)

    print("### Creating model")
    model, tokenizer = create_model(config)

    model.eval()

    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    print("### Creating datasets")
    dataloader = create_dataset(config)

    print("### Start evaluating")

    predictions = evaluation(config, model, dataloader, env)

    utils.write_jsonl(predictions, output_path)
    print("### Prediction Results Save To: ", output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default="")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(args, config)