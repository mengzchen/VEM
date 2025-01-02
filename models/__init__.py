from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def create_model(config):
    if config["model_name"] == "cogagent":
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        return model, tokenizer
    else:
        assert f"not support such model: {config['model_name']}"