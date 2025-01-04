import argparse
import yaml
import json
import os
from accelerate import DistributedDataParallelKwargs, Accelerator

import utils
from dataset import create_dataset
from models import create_agent
from eval_tools.androidenv import AndroidEnv, interact_environment



def evaluation(agent, dataset, env):
    anns = interact_environment(
        agent=agent,
        env=env,
        dataset=dataset
    )
    
    return anns


def main(config):
    print("config:", json.dumps(config))
    output_path = os.path.join("checkpoints/results/", f"{config['model_name']}.jsonl")
    print("output_path: ", output_path)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    print("### build android env")

    env = AndroidEnv(config)

    print("### Creating agent")
    agent = create_agent(config)
    
    print("### Creating datasets")
    dataset = create_dataset(config)

    print("### Start evaluating")

    predictions = evaluation(agent, dataset, env=None)

    utils.write_jsonl(predictions, output_path)
    print("### Prediction Results Save To: ", output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default="")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)