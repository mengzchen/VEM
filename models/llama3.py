import torch
from transformers import AutoTokenizer, AutoModelForCausalLM




model_dir = ""
model = AutoModelForCausalLM.from_pretrained(model_dir, use_cache=False, torch_dtype = torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_cache=False)
        

