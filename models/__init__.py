import gradio_client
from gradio_client import Client, file


class CogAgent:
    def __init__(self, config):
        self.client = Client(config["agent_url"])
    
    def get_action(self, observation):
        text = f'What steps do I need to take to "{observation['task']}"?(with grounding)'
        out = self.client.predict(history=[text], img_path=observation["image_path"], api_name="/predict")
        return out


def create_agent(config):
    if config["model_name"] == "cogagent":
        return CogAgent(config)
    # elif config["model_name"] == "autoui":
    #     from models.autoui_agent import AutoUIAgent
    #     return AutoUIAgent
    else:
        assert f"not support such model: {config['model_name']}"

test_client = CogAgent(config={"agent_url": "https://325578253987c92323.gradio.live"})
test_client.get_action(observation={"task": "search the weather in Beijing", "image_path": "images/aitw_images/general/56622825824867794_0.png"})