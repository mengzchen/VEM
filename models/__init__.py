from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CogAgent:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        

    def get_action(self):
        pass


def create_agent(config):
    if config["model_name"] == "cogagent":
        return CogAgent(config)
    else:
        assert f"not support such model: {config['model_name']}"