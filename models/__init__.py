from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def create_model(config):
    if config["model_name"] == "cogagent":
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            torch_dtype=torch.bfloat16
        )
        return model, tokenizer
    else:
        assert f"not support such model: {config['model_name']}"