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
        self.model = AutoModelForCausalLM.from_pretrained(config["policy_lm"], trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(config["qwen_path"], trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(config["qwen_path"], trust_remote_code=True)
        
        print(f"### load critic: {config['critic_lm']}")
        self.critic = Qwen2VLForConditionalGeneration.from_pretrained(config['critic_lm'], torch_dtype=torch.bfloat16).to("cuda:1")
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
        total_loss = 0
        for text, image_path, target in zip(texts, image_paths, targets):
            target_id = self.tokenizer(target).to(self.model.device)
            print(target_id)
            exit()
            query = self.tokenizer.from_list_format([{'image': image_path}, {'text': text}])
            logits = self.model.get_logits(self.tokenizer, query=query, history=None)
            
            prediction_probs = self.softmax(logits)
            selected_prediction_probs = torch.take_along_dim(prediction_probs, target_id["input_ids"].unsqueeze(2), dim=2).squeeze(2)
            selected_prediction_probs = torch.clamp(selected_prediction_probs, min=0.001, max=0.99)
            
            loss = torch.log(selected_prediction_probs)
            total_loss += loss.sum() 
        return total_loss
    
    
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