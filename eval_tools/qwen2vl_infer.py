from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics import classification_report

import utils
from tqdm import tqdm
import os


class Infer:
    def __init__(self, model_name, test_name):
        self.model_path = f"checkpoints/{model_name}"
        os.system(f"cp checkpoints/Qwen2-VL-7B-Instruct/chat_template.json {self.model_path}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.test_name = test_name

        if test_name == "aitw":
            test_rpath = "data/aitw_anns/1102/aitw_train_label_v5_post_v2_val.json"
        elif test_name == "ufo":
            test_rpath = "data/ufo_anns/1105/ufo_test.json"
        else:
            utils.colorful_print(f"not such test_name: {test_name}", "red")
        
        self.anns = utils.read_json(test_rpath)

        utils.colorful_print(f"### len of test anns: {len(self.anns)}", "green")
    
    def get_input(self, ann):
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": ann["messages"][0]["content"].replace("<image>\n", "")},
                {"type": "image", "image": ann["images"][0]}
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
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    def compute_metrix(self, results):
        if self.test_name == "ufo":
            from eval_tools.ufo import compute_ufo
            compute_ufo(results)
        elif self.test_name == "aitw":
            y_true, y_pred = [], []
            for result in results:
                y_true.append(result["rating"])
                if "Rating" in result["prediction"]:
                    result["response"] = result["prediction"]
                    rating = utils.parse_rating(result)
                    y_pred.append(rating)
                else:
                    y_pred.append(int(result["prediction"]))
            print(classification_report(y_true, y_pred, zero_division=1))
        else:
            pass

    def infer_all(self):
        results = []
        for ann in tqdm(self.anns):
            inputs = self.get_input(ann)
            output = self.infer_one(inputs)
            ann["prediction"] = str(output)
            results.append(ann)
        
        utils.write_jsonl(results, f"{self.model_path}/results.jsonl")
        self.compute_metrix(results)

Infer(model_name="qwen2vl_ufo_1105_step_1500", test_name="ufo").infer_all()