from data_preprocess.cloudgpt_aoai import get_chat_completion, encode_image
import utils

prompt_v1 = """
You are an expert in verifying GUI actions against task requirements. Your goal is to evaluate the provided action and its corresponding screenshot. Follow these guidelines:

1. Analyze the provided screenshot.
2. Compare the action with the task description: if complete, return True; otherwise, return False.
3. Assign a score (between 0 and 10) to the current action based on its accuracy and completeness.
4. Return the result in the following JSON format: {"step score": int, "status": bool}

Example input:
    Task: Open the files app
    Current Action: {"action_type": status task complete}
Example output: {"step score": 9, "status": True}

"""


prompt_v2 = """
You're an expert in evaluating whether the Screenshot successfully completes the Task.

Example:
Task: What's the news in French today?
Q: What should I expect to see on the screenshot if I've searched for the news in French today?
A: I should expect to see some news in French today, such as someone did something or some accident happens in French today. The screenshot shows I'm in the website france24.com and can see the news, like something about the Olympic flame.
Status: success

Task: {}
Respond in this format:
Q: What should I expect to see on the screenshot if I've <repeat the task>?
A: I should expect to see <first expectation, then what's in the given screenshot.>
Status: success or failure (don't return anything else)
Start with "Q:"
"""


prompt_v3 = """
Task: Evaluate the next predicted action based on the given GUI screenshot and task requirements.

Instructions:
1. Review the GUI Screenshot: Carefully examine the provided screenshot to understand the current state of the user interface.
2. Assess the Task Requirements: Understand what the task is asking for. Make note of any specific actions that need to be taken or avoided.
3. Evaluate the Next Predicted Action: Determine whether the predicted next action aligns with the task requirements. Consider the following:
   - Does the action progress the task effectively?
   - Is the action likely to navigate to an unwanted page, such as an advertisement?
4. Rate the Action: Provide a rating from 0 to 10.
   - Scores should be based on how well the action aligns with the task requirements.
   - Actions leading to ads or other distractions should receive lower scores.
5. Justify the Rating: Offer a brief explanation for the score assigned.

Output Format:
Rating: [0-10]
Evaluation: [Detailed explanation of why the score was assigned, referencing specific elements from the GUI screenshot and task requirements. Explain if the action is productive or if it leads to distractions like advertisements.]

Example Input:
Task Requirements: "User needs to navigate to the settings page and enable Dark Mode."
Next Predicted Action: "Click on the 'Offers' button."

Example Output:
Rating: 2
Evaluation: The predicted next action is clicking on the 'Offers' button, which is not related to the task requirement of navigating to the settings page to enable Dark Mode. Clicking the 'Offers' button is likely to lead to an advertisement page, distracting from the task at hand. The appropriate action should involve finding and clicking on the 'Settings' button.

Task Requirements: {}
Next Predicted Action: {}
"""


prompt_v4 = """
Task: Evaluate the prediction of next action based on the given GUI screenshot and task requirements.

Instructions:
1. Review the GUI Screenshot: Carefully examine the provided screenshot to understand the current state of the user interface. If the predicted action is click, there will be a red bbox on the screenshot.
2. Assess the Task Requirements: Understand what the task is asking for. 
3. Evaluate the prediction of Next Action: base on the Given history action and next action, judging if it matches the need of task.
4. Rate the Action: Provide a rating based on four levels:
   - Level 1: The action is highly likely to lead to an advertisement or a completely irrelevant page, deviating significantly from the task requirements.
   - Level 2: The action might lead to a non-ideal page but is somewhat aligned with the task direction.
   - Level 3: The action contributes to the task requirements but might not be the most efficient path.
   - Level 4: The action is the optimal step towards achieving the task requirements.
5. Justify the Rating: Offer a brief explanation for the score assigned.

Explanation of Action:
- `action_type`: including 'click', 'scroll down', 'scroll up', 'status task complete'(means the task is ended), 'press home'(return to the home page), 'press back', 'type'(typing text)
- `click_point`: if the action_type is click, this key will provide the relative position of the screenshot, (x, y) between [0, 1]
- `typed_text`: if the action_type is type, this key will provide the content.

Output Format:
Rating: [1-4]
Evaluation: [Detailed explanation of why the score was assigned]

Example Input:
Task Requirements: "User needs to navigate to the settings page and enable Dark Mode."
History Action: 
Step 0: action_type is press home.
Next Predicted Action: action_type is click, click_point is (0.1, 0.23).

Example Output:
Rating: 2
Evaluation: The predicted next action is clicking on the 'Offers' button, which is not related to the task requirement of navigating to the settings page to enable Dark Mode. Clicking the 'Offers' button is likely to lead to an advertisement page, distracting from the task at hand. The appropriate action should involve finding and clicking on the 'Settings' button.

Task Requirements: {}
History Action: {}
Prediction of Next Action: {}
"""


prompt_v5 = """
Task: Evaluate the prediction of Current action based on the given GUI screenshot and task requirements.

Instructions:
1. Review the Entire GUI Screenshot Sequence: Carefully examine the provided sequence of screenshots to understand the full sequence of actions and the current state of the user interface. If the predicted action involves a click, there will be a red point on the screenshot.
2. Assess the Task Requirements: Understand what the task is asking for.
3. Evaluate the Prediction of Current Action: Based on the provided history of actions and the predicted Current action, determine if it aligns with the task requirements.
4. Rate the Action: Provide a rating based on four levels:
   - Level 1: The action is highly likely to lead to an advertisement or a completely irrelevant page, significantly deviating from the task requirements.
   - Level 2: The action might lead to a non-ideal page but is somewhat aligned with the task direction.
   - Level 3: The action contributes to the task requirements but may not be the most efficient path.
   - Level 4: The action is the optimal step towards achieving the task requirements.     
5. Justify the Rating: Offer a brief explanation for the score assigned.

Explanation of Action:
- `action_type`: includes 'click', 'scroll down', 'scroll up', 'status task complete' (indicates the task is completed), 'press home' (return to the home page), 'press back', 'type' (typing text)
- `click_point`: if the action_type is click, this key will provide the relative position on the screenshot, (x, y) between [0, 1]
- `typed_text`: if the action_type is type, this key will provide the content.

Output Format:
Rating: [1-4]
Evaluation: [Detailed explanation of why the score was assigned]

Example Input:
Task Requirements: User needs to navigate to the settings page and enable Dark Mode. 
Action and ScreenShot:
1. Step 0: action_type is press home.
2. Step 1: action_type is click, click_point is (0.3, 0.8).
3. Step 2: action_type is status task complete.
Current Action: Step 1: action_type is click, click_point is (0.3, 0.8).

Example Output:
Rating: 2
Evaluation: The predicted next action is clicking on the 'Offers' button, which is not related to the task requirement of navigating to the settings page to enable Dark Mode. Clicking the 'Offers' button is likely to lead to an advertisement page, distracting from the task at hand. The appropriate action should involve finding and clicking on the 'Settings' button.

Task Requirements: {}
Action and ScreenShot: 
{}
Current Action: {}
"""


def get_message(text_list, image_path_list) -> list:
    if len(image_path_list) == 0:
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ]
        }]
    else:
        assert len(text_list) == len(image_path_list), f"text len: {len(text_list)} image len: {len(image_path_list)}"
        content = []
        for (text, image) in zip(text_list, image_path_list):
            image = encode_image(image)
            content.append({"type": "text", "text": text})
            content.append({"type": "image_url", "image_url": {"url": image}})
        message = [{
            "role": "user",
            "content": content
        }]

    return message
    

class GPTScorer:
    def __init__(self, version: str):
        self.version = version
        if version == "v1":
            self.prompt = prompt_v1
        elif version == "v2":
            self.prompt = prompt_v2
        elif version == "v3":
            self.prompt = prompt_v3
        elif version == "v4":
            self.prompt = prompt_v4
        elif version == "v5":
            self.prompt = prompt_v5
        else:
            ValueError
    
    def get_label_anns(self, ann_rpath: str, ann_wpath: str):
        anns = utils.read_json(ann_rpath)[:200]

        label_anns = []
        for ann in anns:
            ann["response"] = self.get_one_answer(ann)
            label_anns.append(ann)
        
        label_anns = utils.parse_rating(label_anns)
        utils.write_json(label_anns, ann_wpath)
        utils.write_to_excel(label_anns, f"{self.version}.xlsx")

    def get_one_answer(self, step: dict) -> str:
        task_descibe = self.prompt.format(step["task"], "\n".join(step["action_list"]), step["action"])
        
        # split the task, there is '\n' at -1
        task_descibe = task_descibe.split("<image>")[:-1]
        message = get_message(task_descibe, step["image_path_list"] + [step["image_path"]])
        
        response = get_chat_completion(
            engine="gpt-4o-20240513",
            messages=message,
        )

        print(response.choices[0].message.content)
        utils.colorful_print("-" * 50, "blue")

        return response.choices[0].message.content

    def use_gpt(self, text: str, image_path: str) -> str:
        message = get_message(text_list=[text], image_path_list=image_path)
        response = get_chat_completion(
            engine="gpt-4o-20240513",
            messages=message,
        )
        print(response.choices[0].message.content)


version = "v5"
gptscorer = GPTScorer(version=version)

# text = """
# """
# gptscorer.use_gpt(text=text, image_path=[])

ann_rpath = "data/aitw_train_1017.json"
ann_wpath = f"data/aitw_train_1017_label_{version}.json"
gptscorer.get_label_anns(ann_rpath=ann_rpath, ann_wpath=ann_wpath)

