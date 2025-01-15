import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration, AutoModelForCausalLM
from PIL import Image
from data_preprocess.utils import cogagent_translate_action


class CogAgent(torch.nn.Module):
    def __init__(self, accelerator, config):
        super(CogAgent, self).__init__()
        print(f"### load policy: {config['policy_lm']}")
        self.tokenizer = AutoTokenizer.from_pretrained(config["policy_lm"], padding_side="left", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(config["policy_lm"], torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda") 

        print(f"### load critic: {config['critic_lm']}")
        # TODO freeze the critic
        self.critic = Qwen2VLForConditionalGeneration.from_pretrained(config['critic_lm'], torch_dtype=torch.bfloat16)
        self.critic_processor = AutoProcessor.from_pretrained(config['critic_lm'])

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
        image = Image.open(image_path)
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
            outputs = outputs[:, inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return response
        
    def get_action(self, texts, image_paths):
        results = []
        for text, image_path in zip(texts, image_paths):
            results.append(self._get_action(text, image_path))
        
        return results
