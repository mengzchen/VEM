import torch
from PIL import Image 
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import spaces


@spaces.GPU()
def predict(text, image_path):
    query = tokenizer.from_list_format([{'image': image_path}, {'text': text}])
    with torch.no_grad():
        response, _ = model.chat(tokenizer, query=query, history=None)
        return response
    

def main():
    global tokenizer, model
    model_dir = "checkpoints/SeeClick-aitw"
    qwen_path = "checkpoints/Qwen-VL-Chat"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, padding_side="right", use_fast=False,trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)

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
