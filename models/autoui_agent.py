import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from models.T5_model import T5ForMultimodalGeneration
import utils


class AutoUIAgent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm, critic_lm, do_sample, temperature, max_new_tokens):
        super(AutoUIAgent, self).__init__()

        utils.colorful_print(f"### load policy lm: {policy_lm}", "green")
        self.model = T5ForMultimodalGeneration.from_pretrained(policy_lm, torch_dtype = torch.bfloat16).to(device)
        
        utils.colorful_print(f"### load critic && trajectory critic: {critic_lm}", "green")
        self.critic = Qwen2VLForConditionalGeneration.from_pretrained(critic_lm, torch_dtype=torch.bfloat16).to(device)
        self.critic_processor = AutoProcessor.from_pretrained(critic_lm)

        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=0.5)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
    
    def prepare(self):
        self.model = self.accelerator.prepare(self.model)
        self.critic = self.accelerator.prepare(self.critic)

    def get_action(self, observation, image_features):
        image_features = image_features[..., -1408:]
        
        with torch.no_grad():
            obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
            image_features = image_features.to(self.device)
            outputs = self.accelerator.unwrap_model(self.model).generate(
                **obs_ids, image_ids = image_features,
                max_new_tokens=self.max_new_tokens, 
                do_sample=self.do_sample, 
                temperature = self.temperature,
                pad_token_id = self.tokenizer.eos_token_id
            ).cpu()

        raw_action = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        raw_action = [a[1:] if a.startswith('\n') else a for a in raw_action]

        return raw_action

    def get_log_prob(self, observation, image_features, action):
        image_features = image_features[...,-1408:]
        
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        outputs = self.model(
            input_ids = obs_ids["input_ids"],
            image_ids = image_features,
            attention_mask = obs_ids["attention_mask"],
            labels = action_ids["input_ids"]
        )
        # TODO 
        
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs, action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        selected_prediction_probs = torch.clamp(selected_prediction_probs, min=0.001, max=0.99)
        
        return torch.log(selected_prediction_probs) * action_ids["attention_mask"]