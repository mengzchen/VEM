import os
import sys
sys.path.append(os.getcwd())
import torch
import gradio as gr
from transformers import AutoTokenizer
from models.T5_model import T5ForMultimodalGeneration
import spaces
from models.autoui_agent import ImageFeatureExtractor


@spaces.GPU()
def predict(text, image_path):
    image_features = image_features = torch.stack([image_feature_extractor.to_feat(image_path)[..., -1408:]])
    
    with torch.no_grad():
        text_ids = tokenizer([text], return_tensors='pt', padding=True, max_length=512, truncation=True).to(model.device)
        image_features = image_features.to(model.device, torch.bfloat16)
        
        outputs = model.generate(
            **text_ids, image_ids=image_features, max_new_tokens=256
        )
        raw_actions = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        print(raw_actions)
        
    return raw_actions[0]


def main():
    global tokenizer, model, image_feature_extractor
    model_dir = "checkpoints/Auto-UI-Base"
    model = T5ForMultimodalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, device_map="auto")
    image_feature_extractor = ImageFeatureExtractor()

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.'),
            gr.Image(type="filepath", label="Image Prompt", value=None),
        ],
        outputs="text"
    )

    demo.launch(share=True, show_error=True)


if __name__ == "__main__":
    main()
    # predict("test", "images/aitw_images/general/3266935758626570610_0.png")
