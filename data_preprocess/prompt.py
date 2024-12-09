prompt_give_score = """
Task: Evaluate the prediction of Current action based on the given GUI screenshot and task requirements.

Instructions:
1. Review the Entire GUI Screenshot Sequence: Carefully examine the provided sequence of screenshots to understand the full sequence of actions and the current state of the user interface. If the predicted action involves a click, there will be a red point on the screenshot.
2. Assess the Task Requirements: Understand what the task is asking for.
3. Evaluate the Prediction of Current Action: Based on the provided entire actions and GUI screenshot, determine if current action aligns with the task requirements.
4. Rate the Action: Provide a rating based on four levels:
   - Level 1: The action is significantly deviating from the task requirements.
   - Level 2: The action might lead to a non-ideal page but is somewhat aligned with the task direction.
   - Level 3: The action contributes to the task requirements but may not be the most efficient path.
   - Level 4: The action is the optimal step towards achieving the task requirements.     
5. Justify the Rating: Offer a brief explanation for the score assigned. (explanation can only mention actions had been done.)

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
Step 0: action_type is press home.
Step 1: action_type is click, click_point is (0.3, 0.8).
Step 2: action_type is status task complete.
Current Action: 
Step 1: action_type is click, click_point is (0.3, 0.8).

Example Output:
Rating: 2
Evaluation: The predicted next action is clicking on the 'Offers' button, which is not related to the task requirement of navigating to the settings page to enable Dark Mode. Clicking the 'Offers' button is likely to lead to an advertisement page, distracting from the task at hand. The appropriate action should involve finding and clicking on the 'Settings' button.

Task Requirements: {}
Action and ScreenShot: 
{}
Current Action: 
{}
"""


prompt_critic_input = """Task: Evaluate the prediction of Current action based on the given GUI screenshot and task requirements.

Instructions:
1. Review the history Action: Carefully examine the provided sequence of actions.
2. Assess the Task Requirements: Understand what the task is asking for.
3. Evaluate the Prediction of Current Action: Based on the provided entire actions and GUI screenshot, determine if current action aligns with the task requirements.
4. Rate the Action: Provide a rating based on four levels:
   - Level 1: The action is significantly deviating from the task requirements.
   - Level 2: The action contributes to the task requirements but may not be the most efficient path.
   - Level 3: The action is the optimal step towards achieving the task requirements.     

Explanation of Action:
- `action_type`: includes 'click', 'scroll down', 'scroll up', 'status task complete' (indicates the task is completed), 'press home' (return to the home page), 'press back', 'type' (typing text)
- `click_point`: if the action_type is click, this key will provide the relative position on the screenshot, (x, y) between [0, 1]
- `typed_text`: if the action_type is type, this key will provide the content.

Output Format:
[1-3]

Example Input:
Task Requirements: User needs to navigate to the settings page and enable Dark Mode. 
History Action:
Step 0: action_type is press home.
Current Action: 
Step 1: action_type is click, click_point is (0.3, 0.8).

Example Output:
2

Task Requirements: {}
History Action: 
{}
Current Action: 
{}
"""
