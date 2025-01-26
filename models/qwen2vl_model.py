import torch
import transformers
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from transformers.generation import GenerationConfig
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model
from peft import AutoPeftModelForCausalLM


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def preprocess(
    texts,
    image_paths, 
    outputs,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = 512,
    system_message: str = "You are a helpful assistant."
):
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for (text, image_path, output) in zip(texts, image_paths, outputs):
        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        
        # add prompt
        prompt = f"Picture 1: <img>{image_path}</img>\n{text}"
        _input_id = tokenizer(roles["user"]).input_ids + nl_tokens + tokenizer(prompt).input_ids + [im_end] + nl_tokens
        _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
        input_id += _input_id
        target += _target

        # add target
        _input_id = tokenizer(roles["assistant"]).input_ids + nl_tokens + tokenizer(output).input_ids + [im_end] + nl_tokens
        _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(roles["assistant"]).input_ids) + _input_id[len(tokenizer(roles["assistant"]).input_ids) + 1:-2] + [im_end] + nl_tokens
        input_id += _input_id
        target += _target

        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    
    input_ids = torch.tensor(input_ids, device="cuda:0")
    targets = torch.tensor(targets, device="cuda:0")

    return input_ids, targets, input_ids.ne(tokenizer.pad_token_id)


class Qwen2VLAgent(torch.nn.Module):
    def __init__(self, accelerator, config, is_eval):
        super(Qwen2VLAgent, self).__init__()
        if not is_eval:
            print(f"### load policy: {config['policy_lm']}")
            self.model = AutoModelForCausalLM.from_pretrained(
                config["policy_lm"], 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            ).to(config["policy_device"])
            # customized LoRA parameters
            target_modules = []
            target_layer_names = ["visual.conv1", "attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj", "c_attn",
                                "attn.c_proj", "w1", "w2"]
            lora_supported_types = [torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, transformers.pytorch_utils.Conv1D]
            for name, module in self.model.named_modules():
                if any(t_name in name for t_name in target_layer_names) and 'attn_pool' not in name:
                    if isinstance(module, tuple(lora_supported_types)):
                        target_modules.append(name)
                    else:
                        print(name + " not satisfy lora")
                        input()
            modules_to_save = ["wte", "lm_head"]
            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                modules_to_save=modules_to_save  # This argument serves for adding new tokens.
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.enable_input_require_grads()

            self.model.transformer.visual.requires_grad_(False)
            if hasattr(self.model.transformer.visual, 'attn_pool'):
                self.model.transformer.visual.attn_pool.requires_grad_(True)
        else:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                config["save_model"], 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            ).to(config["policy_device"])

        self.tokenizer = AutoTokenizer.from_pretrained(config["qwen_path"], padding_side="right", use_fast=False,trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.model.generation_config = GenerationConfig.from_pretrained(config["qwen_path"], trust_remote_code=True)
        
        print(f"### load critic: {config['critic_lm']}")
        self.critic = Qwen2VLForConditionalGeneration.from_pretrained(config['critic_lm'], torch_dtype=torch.bfloat16).to(config["critic_device"])
        print(f"### {self.model.device} {self.critic.device}")
        
        for param in self.critic.parameters():
            param.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(config['critic_lm'])

        self.dropout = torch.nn.Dropout(p=0.5)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.accelerator = accelerator
    

    def prepare(self):
        self.policy = self.accelerator.prepare(self.model)
        self.critic = self.accelerator.prepare(self.critic)


    def get_log_prob(self, texts, image_paths, targets):
        input_ids, labels, attention_mask = preprocess(texts, image_paths, targets, self.tokenizer)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs, input_ids.unsqueeze(2), dim=2).squeeze(2)
        selected_prediction_probs = torch.clamp(selected_prediction_probs, min=0.001, max=0.99)
        
        return torch.log(selected_prediction_probs) * attention_mask
    
    
    def _get_action(self, text, image_path):
        query = self.tokenizer.from_list_format([{'image': image_path}, {'text': text}])
        with torch.no_grad():
            response, _ = self.model.chat(self.tokenizer, query=query, history=None)
            return response
        

    def get_action(self, texts, image_paths):
        results = []
        for text, image_path in zip(texts, image_paths):
            results.append(self._get_action(text, image_path))
        
        return results