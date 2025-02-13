import torch
from PIL import Image 
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import spaces
from peft import AutoPeftModelForCausalLM
import argparse


@spaces.GPU()
def predict(text, image_path):
    query = tokenizer.from_list_format([{'image': image_path}, {'text': text}])
    with torch.no_grad():
        response, _ = model.chat(tokenizer, query=query, history=None)
        return response
    

def main(save_model):
    global tokenizer, model
    qwen_path = "checkpoints/Qwen-VL-Chat"
    
    if save_model == "":
        model = AutoModelForCausalLM.from_pretrained(
            "./checkpoints/SeeClick-aitw", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to("cuda")
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            save_model, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to("cuda")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)

    args = parser.parse_args()

    if args.task == "general":
        save_model = f"checkpoints/seeclick_general_v3/epoch_0"
    elif args.task == "webshop":
        save_model = f"checkpoints/seeclick_webshop_v3/epoch_1"
    else:
        save_model = ""

    main(save_model)
