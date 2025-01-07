import os
import sys
sys.path.append(os.getcwd())
import torch
import gradio as gr
from accelerate import Accelerator
import spaces
from models.autoui_agent import AutoUIAgent
from train_rl import DigiRLTrainer


@spaces.GPU()
def predict(text, image_path):
    image_features = image_features = torch.stack([trainer.image_process.to_feat(image_path)[..., -1408:]])

    raw_actions = trainer.agent.get_action([text], image_features.to(dtype=torch.bfloat16))
    
    return raw_actions[0]


def main():
    global trainer
    
    accelerator = Accelerator()
    device = accelerator.device

    print("### load AutoUIAgent")

    agent = AutoUIAgent(
        device=device,
        accelerator=accelerator,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=128,
        policy_lm="checkpoints/Auto-UI-Base",
        critic_lm="checkpoints/critic_1218/merge-520",
    )
    trainer = DigiRLTrainer(
        agent=agent,
        accelerator=accelerator,
        tokenizer=agent.tokenizer
    )

    # trainer.load("checkpoints/general-off2on-digirl")
    trainer.load("checkpoints/rl-1227/epoch_13")

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
