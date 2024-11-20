import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.digirl_dataset import DummyDataset
import utils


def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            if "min" in key:
                mean_dict[key] = min(d[key] for d in dict_list)
            elif "max" in key:
                mean_dict[key] = max(d[key] for d in dict_list)
            else:
                mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

class DigiRLTrainer():
    def __init__(self, 
        agent,
        accelerator,
        tokenizer,
        lm_lr: float = 1e-5,
        grad_accum_steps: int = 8,
        gamma: float = 0.9,
        tau: float = 0.1,
        epochs: int = 3,
        max_grad_norm: float=0.01,
        actor_epochs: int = 3,
    ):
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lm_lr)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.grad_accum_steps = grad_accum_steps
        self.gamma = gamma

        self.actor_epochs = actor_epochs
        self.epochs = epochs

        self.step = 0
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.softmax = torch.nn.Softmax(dim = -1)


    def prepare(self):
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)
    

    def actor_loss(
        self,
        observation,
        action, 
        image_features, 
        validation=False,
        **kwargs
    ):
        dtype = self.accelerator.unwrap_model(self.agent.model).dtype
        device = self.accelerator.unwrap_model(self.agent.model).device

        # TODO add qwen2vl format data, use image
        batch_size = len(action)
        example = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Give a random num from [1, 2, 3]"}
            ]
        }]
        messages = [example for _ in range(batch_size)]
        text = self.agent.critic_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.agent.critic_processor(
            text=text,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        generated_ids = self.agent.critic.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # TODO turn str to int and tensor with gradient
        value = self.agent.critic_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # turn to int
        value = [int(val) for val in value]
        value = torch.tensor(value, dtype=dtype, requires_grad=True).to(device)
        
        advantages = torch.clamp(value, -1, 1).flatten()

        image_features = image_features.to(device, dtype=dtype)
        log_prob = self.agent.get_log_prob(observation, image_features, action).sum(dim=1).flatten()
        
        pg_loss = - torch.mean(log_prob * advantages)

        if not validation:
            self.accelerator.backward(pg_loss)

        advantages = advantages.detach().cpu()

        info =  {
            "pg.loss": pg_loss.detach().cpu().item(),
            "advantages.mean": advantages.mean(),
            "advantages.max": torch.max(advantages),
            "advantages.min": torch.min(advantages),
            "advantages.std": torch.std(advantages),
        }
        
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        
        return info
        
        
    def update_policy(self, replay_buffer, validation_buffer=None, no_update_actor=False):
        self.step += 1
        info = {}
        info_list = []
        action_bsize = replay_buffer.batch_size
        
        if not no_update_actor:
            utils.colorful_print(">>>updating actor", "green")
            for _ in tqdm(range(self.actor_epochs), disable=not self.accelerator.is_main_process):
                data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps * replay_buffer.batch_size)]
                
                for d in data:
                    for k, v in d.items():
                        d[k] = v[0]

                dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False)
                
                dataloader = self.accelerator.prepare(dataloader)
                self.lm_optimizer.zero_grad()

                for batch in dataloader:
                    advantages = None
                    info_list.append(self.actor_loss(**batch, advantages=advantages))

                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()

        info.update(dict_mean(info_list))

        if validation_buffer is not None:
            info_list = []
            data = [validation_buffer.sample(1) for _ in range(self.grad_accum_steps * replay_buffer.batch_size)]
            
            for d in data:
                for k,v in d.items():
                    d[k] = v[0]

            dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False)
            dataloader = self.accelerator.prepare(dataloader)

            with torch.no_grad():
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.actor_loss(validation=True, advantage=None, **batch))

            info.update(dict_mean(info_list))
            return info
        
        print(info)
        return info


    def update(self, replay_buffer, validation_buffer=None, no_update_actor=False):
        info = {}
        info.update(self.update_policy(replay_buffer, validation_buffer, no_update_actor=no_update_actor))
        
        return info


    def save(self, path):
        self.accelerator.save_state(path, safe_serialization=False)


    def load(self, path):
        self.accelerator.load_state(path)
