prompt_give_score = """Task: Evaluate the prediction of Current action based on the given GUI screenshot and task requirements.

Instructions:
1. Understand the Task Requirements: Understand what the task is asking for.
2. Review the Entire Screenshots Sequence: Carefully examine the provided sequence of screenshots and actions. If the predicted action involves a click, there will be a red point on the screenshot.
3. Evaluate the Prediction of Current Action: Based on the provided entire actions and GUI screenshot, determine if current action aligns with the task requirements.
4. Rate the Action: 
   - Level 1: The action is significantly deviating from the task requirements.
   - Level 2: The action might lead to a non-ideal page but is somewhat aligned with the task direction. The action contributes to the task requirements but may not be the most efficient path.
   - Level 3: The action is the optimal step towards achieving the task requirements.     
5. Explanation: Offer a brief explanation for the score assigned. (explanation can only mention actions had been done.)

Explanation of Action:
- `action_type`: includes 'DUAL_CLICK', 'TYPE' (typing text), 'STATUS_TASK_COMPLETE' (indicates the task is completed), 'PRESS_HOME' (return to the home page), 'PRESS_BACK', 'STATUS_TASK_IMPOSSIBLE'
- `touch_point` and `lift_point`: provide the relative position on the screenshot, [x, y] between [0, 1]
- `typed_text`: if the action_type is type, this key will provide the content.

Output Format:
Rating: [1-3]
Evaluation: [explanation of why the score was assigned]

Example Input:
Task Requirements: What is the capital of England? 
Action and ScreenShot:
step 0: "action_type": "DUAL_POINT", "touch_point": "[0.524, 0.06]", "lift_point": "[0.524, 0.06]", "typed_text": "" 
step 1: "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "capital of England"
step 2: "action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""
step 3: "action_type": "STATUS_TASK_COMPLETE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""
Current Action: 
step 2: "action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""

Example Output:
Rating: 3
Evaluation: The action of pressing Enter after typing "capital of England" is an appropriate step to try to get the answer for the task requirement of finding out what the capital of England is, which is an optimal action towards achieving the task goal.

Task Requirements: {}
Action and ScreenShot: 
{}
Current Action: 
{}
"""


prompt_critic_input = """Task: Evaluate the prediction of Current action based on the given GUI screenshot and task requirements.

Instructions:
1. Understand the Task Requirements: Understand what the task is asking for.
2. Review the Current Screenshots: Carefully examine the provided sequence of screenshots and actions. If the predicted action involves a click, there will be a red point on the screenshot.
3. Evaluate the Prediction of Current Action: Based on the provided history actions and GUI screenshot, determine if current action aligns with the task requirements.
4. Rate the Action:
   - Level 1: The action is significantly deviating from the task requirements.
   - Level 2: The action might lead to a non-ideal page but is somewhat aligned with the task direction. The action contributes to the task requirements but may not be the most efficient path.
   - Level 3: The action is the optimal step towards achieving the task requirements.     

Explanation of Action:
- `action_type`: includes 'DUAL_CLICK', 'TYPE' (typing text), 'STATUS_TASK_COMPLETE' (indicates the task is completed), 'PRESS_HOME' (return to the home page), 'PRESS_BACK', 'STATUS_TASK_IMPOSSIBLE'
- `touch_point` and `lift_point`: provide the relative position on the screenshot, [x, y] between [0, 1]
- `typed_text`: if the action_type is type, this key will provide the content.

Output Format:
Rating: [1-3]

Example Input:
Task Requirements: What is the capital of England? 
Action and ScreenShot:
step 0: "action_type": "DUAL_POINT", "touch_point": "[0.524, 0.06]", "lift_point": "[0.524, 0.06]", "typed_text": "" 
step 1: "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "capital of England"
Current Action: 
step 2: "action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""

Example Output:
Rating: 3

Task Requirements: {}
Action and ScreenShot: 
<image>
{}
Current Action: 
{}
"""


prompt_gen_1_action = """
## Task: 
Given the task requirements, a sequence of actions, and their corresponding screenshots, generate a new action that matches a rating of 1 (indicating the action significantly deviates from the task requirements).

## Instructions:
1. Understand the Task Requirements.
2. Review the provided screenshots and actions. Red points on the screenshots indicate clicks.
3. Generate the new action that matches a rating of 1.
4. follow the output format exactly, without adding extra text.

## Action Format:
- `action_type`: Possible values include 'DUAL_CLICK', 'TYPE', 'STATUS_TASK_COMPLETE', 'PRESS_HOME', 'PRESS_BACK', 'STATUS_TASK_IMPOSSIBLE'
- `touch_point` and `lift_point`: Relative positions on the screenshot, specified as [x, y] within the range [0, 1]
- `typed_text`: Content for 'TYPE' actions

Example Input:
Task Requirements: What is the capital of England? 
Action and ScreenShot:
step 0: "action_type": "DUAL_POINT", "touch_point": "[0.524, 0.06]", "lift_point": "[0.524, 0.06]", "typed_text": "" 
step 1: "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "capital of England"
Origin Action: 
step 2: "action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""

Example Output:
"action_type": "STATUS_TASK_COMPLETE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""

Task Requirements: {}
Action and ScreenShot: 
{}
Origin Action: 
{}
"""


prompt_gen_2_action = """
## Task: 
Given the task requirements, a sequence of actions, and their corresponding screenshots, generate a new action that matches a rating of 2 (indicating the action might lead to a non-ideal page but is somewhat aligned with the task direction. The action contributes to the task requirements but may not be the most efficient path.).

## Instructions:
1. Understand the Task Requirements.
2. Review the provided screenshots and actions. Red points on the screenshots indicate clicks.
3. Generate the new action that matches a rating of 2.

## Action Format:
- `action_type`: Possible values include 'DUAL_CLICK', 'TYPE', 'STATUS_TASK_COMPLETE', 'PRESS_HOME', 'PRESS_BACK', 'STATUS_TASK_IMPOSSIBLE'
- `touch_point` and `lift_point`: Relative positions on the screenshot, specified as [x, y] within the range [0, 1]
- `typed_text`: Content for 'TYPE' actions

Example Input:
Task Requirements: What is the capital of England? 
Action and ScreenShot:
step 0: "action_type": "DUAL_POINT", "touch_point": "[0.524, 0.06]", "lift_point": "[0.524, 0.06]", "typed_text": "" 
step 1: "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "capital of England"
Origin Action: 
step 2: "action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""

Example Output:
"action_type": "PRESS_HOME", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""

Task Requirements: {}
Action and ScreenShot: 
{}
Origin Action: 
{}
"""