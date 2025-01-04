import argparse
import os
import re
import torch
from threading import Thread, Event
from PIL import Image, ImageDraw
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
from typing import List
import spaces

stop_event = Event()

def draw_boxes_on_image(image: Image.Image, boxes: List[List[float]], save_path: str):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x_min = int(box[0] * image.width)
        y_min = int(box[1] * image.height)
        x_max = int(box[2] * image.width)
        y_max = int(box[3] * image.height)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    image.save(save_path)


@spaces.GPU()
def predict(text, image):
    stop_event.clear()
    print(image)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": text}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    
    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "position_ids": inputs["position_ids"],
        "images": inputs["images"],
        "max_length": 256,
        "do_sample": True,
        "top_k": 1,
    }
    response = model.generate(generate_kwargs)
    
    return response


def main():
    global tokenizer, model
    model_dir = "checkpoints/cogagent-9b-20241220"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto").eval()

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
