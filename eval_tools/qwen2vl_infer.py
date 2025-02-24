import os
import sys
sys.path.append(os.getcwd())
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics import classification_report

import utils
from tqdm import tqdm
import os


class Infer:
    def __init__(self, model_name, step):
        self.model_path = f"checkpoints/{model_name}/merge-{step}"
        os.system(f"cp checkpoints/Qwen2-VL-7B-Instruct/chat_template.json {self.model_path}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_path, max_pixels=480*28*28)

        if "shopping" in model_name:
            test_rpath = "data/aitw_anns/0108/webshopping_val_critic.jsonl"
        else:
            test_rpath = "data/aitw_anns/0108/general_val_critic.jsonl"
        
        self.anns = utils.read_jsonl(test_rpath)

        print(f"### len of test anns: {len(self.anns)}")
    
    def get_input(self, ann):
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": ann["critic_inputs"][0]["value"]},
                {"type": "image", "image": ann["critic_images"]}
            ]
        }]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        return inputs
    

    def infer_one(self, inputs):
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    def compute_metrix(self, results):
        y_true, y_pred = [], []
        for result in results:
            if "rating" in result.keys():
                y_true.append(result["rating"])
            else:
                y_true.append(result["critic_output"])
            y_pred.append(result["prediction"])
            
        print(classification_report(y_true, y_pred, zero_division=1))

    def infer_all(self):
        if os.path.exists(f"{self.model_path}/results.jsonl"):
            results = utils.read_jsonl(f"{self.model_path}/results.jsonl")
            self.compute_metrix(results)
            return
        
        results = []
        for ann in tqdm(self.anns):
            inputs = self.get_input(ann)
            output = self.infer_one(inputs)
            ann["prediction"] = int(output)
            results.append(ann)
        
        utils.write_jsonl(results, f"{self.model_path}/results.jsonl")
        self.compute_metrix(results)

    def get_case(self, text, image):
        ann = {
            "critic_inputs": [{"value": text}],
            "critic_images": image
        }
        inputs = self.get_input(ann)
        result = self.infer_one(inputs)
        print(result)


# from data_preprocess.prompt import prompt_critic_system, prompt_critic_user
# task = "Show the shopping cart on newegg.com."
# history = """step 0: "action_type": "DUAL_POINT", "click_point": "[0.69,0.79]"
# step 1: "action_type": "DUAL_POINT", "click_point": "[0.43,0.07]"
# step 2: "action_type": "TYPE", "typed_text": "newegg.com" 
# """
# action = "step 3: \"action_type\": \"DUAL_POINT\", \"click_point\": \"[0.19,0.16]\""
# # action = "step 3: \"action_type\": \"PRESS_HOME\""
# text = prompt_critic_system + prompt_critic_user.format(task, history, action)
# print(text)
# image = "test.png"
# Infer(model_name="critic_shopping", step="970").get_case(text, image)
critic = Infer(model_name="critic_general_qwen2.5", step="2780").infer_all()
