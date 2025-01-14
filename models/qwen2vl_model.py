import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
from PIL import Image
from qwen_vl_utils import process_vision_info


class Qwen2VLAgent(torch.nn.Module):
    def __init__(self, accelerator, config):
        super(Qwen2VLAgent, self).__init__()
        print(f"### load policy: {config['policy_lm']}")
        self.model = AutoModelForCausalLM.from_pretrained(config["policy_lm"], trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(config["qwen_path"], trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(config["qwen_path"], trust_remote_code=True)
        
        # TODO freeze the critic
        print(f"### load critic: {config['critic_lm']}")
        self.critic = Qwen2VLForConditionalGeneration.from_pretrained(config['critic_lm'], torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(config['critic_lm'])

        self.dropout = torch.nn.Dropout(p=0.5)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.accelerator = accelerator
    

    def prepare(self):
        self.policy = self.accelerator.prepare(self.policy)
        self.critic = self.accelerator.prepare(self.critic)


    def get_log_prob(self, text, image, target):
        text_ids = self.tokenizer(text, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        target_ids = self.tokenizer(target, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        outputs = self.model(
            input_ids = text_ids["input_ids"],
            image_ids = image,
            attention_mask = text_ids["attention_mask"],
            labels = target_ids["input_ids"]
        )
        
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs, target_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        selected_prediction_probs = torch.clamp(selected_prediction_probs, min=0.001, max=0.99)
        
        return torch.log(selected_prediction_probs) * target_ids["attention_mask"]
    
    
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