test ="""
"""
prompt_score_system ="""As an expert in the field of GUI and reinforcement learning, you will receive complete screenshots and textual descriptions of interactions for a given task. You need to evaluate a specific step in terms of its value within the task chain, similar to what a value function does in reinforcement learning. Detailed criteria and standards are given below.

## Explanation of the input content:
1. Task: Brief description of the current GUI task, such as implementing the "Get Hong Kong hotel prices" task in Android GUI.
2. Complete operation description and corresponding screenshot sequence for the task
   (1) Text description of operations: Contains 11 types of GUI operations. Specific fields and their meanings are as follows:
      [1] DUAL_POINT: Double-click on a specific position on the screen. If it is a link or software, it will enter; if it is text, it will be selected. The "click_point" is represented by a two-dimensional array indicating the position of the click, relative to the top-left corner of the screenshot and within a range from 0.0 to 1.0.
         - example: "action_type": "DUAL_POINT", "click_point": [0.5, 0.5]
      [2] TYPE: An action type that sends text. Note that this simply sends text and does not perform any clicks for element focus or enter presses for submitting text.
         - example: "action_type": "TYPE", "typed_text": "capital of England"
      [3] PRESS_BACK: Return to the previous page. Usually the previous webpage.
         - example: "action_type": "PRESS_BACK"
      [4] PRESS_HOME: Return to the system home page. Use this action to return to the home screen when the current screen is not the desired one, so you can reselect the program you need to enter.
         - example: "action_type": "PRESS_HOME"
      [5] PRESS_ENTER: Press the enter key to execute a step. Generally, after confirming the input text, use this action to start the search.
         - example: "action_type": "PRESS_ENTER"
      [6] STATUS_TASK_COMPLETE: An action used to indicate that the desired task has been completed and resets the environment. This action should also be used if the task is already completed and there is nothing more to do. For example, the task is to turn on the Wi-Fi when it is already on.
         - example: "action_type": "STATUS_TASK_COMPLETE"
      [7] STATUS_TASK_IMPOSSIBLE: An action used to indicate that the desired task is impossible to complete and resets the environment. This can result from various reasons including UI changes, Android version differences, etc.
         - example: "action_type": "STATUS_TASK_IMPOSSIBLE"
      [8] SCROLL_DOWN: Scroll down.
         - example: "action_type": "SCROLL_DOWN"
      [9] SCROLL_UP: Scroll up.
         - example: "action_type": "SCROLL_UP"
      [10] SCROLL_LEFT: Scroll left.
         - example: "action_type": "SCROLL_LEFT"
      [11] SCROLL_RIGHT: Scroll right.
         - example: "action_type": "SCROLL_RIGHT"
   (2) Corresponding screenshot before each operation. If the operation is of the "DUAL_POINT" type, the click position is marked with a red dot in the image.    
3. The current action to be evaluated and the corresponding screenshot.

## Evaluation Criteria:
Here are the detailed descriptions of the two levels. Attention needs to be paid to whether the action taken based on the current screenshot promotes efficient task execution, rather than the relevance of the content shown in the current screenshot to the task:
   Level 1: The action is not the optimal choice for completing the task at this moment, which may lead to deviations from the task flow. For example:
      (1) Incorrect text input.
      (2) Clicking a button that might lead to an advertisement.
      (3) Announcing the task's success when it has not actually been achieved.
   Level 2: The action is the optimal and correct choice for completing the task at this moment. For example:
      (1) When showing task completion, the displayed content can fully achieve it.
      (2) When entering an unrelated interface, you can return to the main screen by executing "PRESS_HOME."
      (3) Selecting the most correct entry point to complete the current task.

## Output requirements:
- Format: {"rating": int, "explanation": str}. Do not include any additional characters beyond this format
- The "rating" field should be represented by the number 1 or 2 indicating the evaluation level. The "explanation" field should explain the evaluation process that led to this rating, without including descriptions of operations after the current step (future operations are considered unknown).

## Example Input:
Task Requirements: What is the capital of England?
Action and ScreenShot:
step 0: "action_type": "DUAL_POINT", "click_point": "[0.524, 0.06]"
step 1: "action_type": "TYPE", "typed_text": "capital of England"
step 2: "action_type": "PRESS_ENTER"
step 3: "action_type": "STATUS_TASK_COMPLETE"
Current Action:
step 2: "action_type": "PRESS_ENTER"

## Example Output:
{"rating": 2, "explanation": "The action of pressing enter after typing 'capital of England' is an appropriate step to get the answer to the task requirement of finding out the capital of England, which is an optimal action towards achieving the task goal."}

"""


prompt_score_user = """Task Requirements: {}
Action and ScreenShot: {}
Current Action: 
{}
"""


prompt_critic_system = """As an expert in the field of GUI and reinforcement learning, you will receive textual descriptions of history interactions for a given task. You need to evaluate the current action, similar to what a value function does in reinforcement learning. Detailed criteria and standards are given below.

## Explanation of the input content:
1. Task: Brief description of the current GUI task, such as implementing the "Get Hong Kong hotel prices" task in Android GUI.
2. Description of History operation
   Contains 11 types of GUI operations. Specific fields and their meanings are as follows:
   [1] DUAL_POINT: Double-click on a specific position on the screen. If it is a link or software, it will enter; if it is text, it will be selected. The "click_point" is represented by a two-dimensional array indicating the position of the click, relative to the top-left corner of the screenshot and within a range from 0.0 to 1.0.
      - example: "action_type": "DUAL_POINT", "click_point": [0.5, 0.5]
   [2] TYPE: An action type that sends text. Note that this simply sends text and does not perform any clicks for element focus or enter presses for submitting text.
      - example: "action_type": "TYPE", "typed_text": "capital of England"
   [3] PRESS_BACK: Return to the previous page. Usually the previous webpage.
      - example: "action_type": "PRESS_BACK"
   [4] PRESS_HOME: Return to the system home page. Use this action to return to the home screen when the current screen is not the desired one, so you can reselect the program you need to enter.
      - example: "action_type": "PRESS_HOME"
   [5] PRESS_ENTER: Press the enter key to execute a step. Generally, after confirming the input text, use this action to start the search.
      - example: "action_type": "PRESS_ENTER"
   [6] STATUS_TASK_COMPLETE: An action used to indicate that the desired task has been completed and resets the environment. This action should also be used if the task is already completed and there is nothing more to do. For example, the task is to turn on the Wi-Fi when it is already on.
      - example: "action_type": "STATUS_TASK_COMPLETE"
   [7] STATUS_TASK_IMPOSSIBLE: An action used to indicate that the desired task is impossible to complete and resets the environment. This can result from various reasons including UI changes, Android version differences, etc.
      - example: "action_type": "STATUS_TASK_IMPOSSIBLE"
   [8] SCROLL_DOWN: Scroll down.
      - example: "action_type": "SCROLL_DOWN"
   [9] SCROLL_UP: Scroll up.
      - example: "action_type": "SCROLL_UP"
   [10] SCROLL_LEFT: Scroll left.
      - example: "action_type": "SCROLL_LEFT"
   [11] SCROLL_RIGHT: Scroll right.
      - example: "action_type": "SCROLL_RIGHT"
3. The current action to be evaluated and the corresponding screenshot(the screenshot before each operation. If the operation is of the "DUAL_POINT" type, the click position is marked with a red dot in the image.)

## Evaluation Criteria:
Here are the detailed descriptions of the two levels. Attention needs to be paid to whether the action taken based on the current screenshot promotes efficient task execution, rather than the relevance of the content shown in the current screenshot to the task:
   Level 1: The action is not the optimal choice for completing the task at this moment, which may lead to deviations from the task flow. For example:
      (1) Incorrect text input.
      (2) Clicking a button that might lead to an advertisement.
      (3) Announcing the task's success when it has not actually been achieved.
   Level 2: The action is the optimal and correct choice for completing the task at this moment. For example:
      (1) When showing task completion, the displayed content can fully achieve it.
      (2) When entering an unrelated interface, you can return to the main screen by executing "PRESS_HOME."
      (3) Selecting the most correct entry point to complete the current task.

## Output requirements: 1 or 2 (INT)

## Example Input:
Task Requirements: What is the capital of England?
Previous Action:
step 0: "action_type": "DUAL_POINT", "click_point": "[0.524, 0.06]"
step 1: "action_type": "TYPE", "typed_text": "capital of England"
Current Action and Screenshot:
step 2: "action_type": "PRESS_ENTER"

## Example Output:
2

"""


prompt_critic_user = """Task Requirements: {}
Previous Action: 
{}
Current Action and Screenshot: 
{}
"""


prompt_negative_system = """As an expert in the field of GUI and 负样本数据构造者, 你需要根据历史的screenshot及对应的action description,任务描述和原始的current action来生成一个新的负样本的current action. Detailed criteria and standards are given below.

## Explanation of the input content:
1. Task: Brief description of the current GUI task, such as implementing the "Get Hong Kong hotel prices" task in Android GUI.
2. History operation description and corresponding screenshot sequence for the task
   (1) Text description of operations: Contains 11 types of GUI operations. Specific fields and their meanings are as follows:
      [1] DUAL_POINT: Double-click on a specific position on the screen. If it is a link or software, it will enter; if it is text, it will be selected. The "click_point" is represented by a two-dimensional array indicating the position of the click, relative to the top-left corner of the screenshot and within a range from 0.0 to 1.0.
         - example: "action_type": "DUAL_POINT", "click_point": [0.5, 0.5]
      [2] TYPE: An action type that sends text. Note that this simply sends text and does not perform any clicks for element focus or enter presses for submitting text.
         - example: "action_type": "TYPE", "typed_text": "capital of England"
      [3] PRESS_BACK: Return to the previous page. Usually the previous webpage.
         - example: "action_type": "PRESS_BACK"
      [4] PRESS_HOME: Return to the system home page. Use this action to return to the home screen when the current screen is not the desired one, so you can reselect the program you need to enter.
         - example: "action_type": "PRESS_HOME"
      [5] PRESS_ENTER: Press the enter key to execute a step. Generally, after confirming the input text, use this action to start the search.
         - example: "action_type": "PRESS_ENTER"
      [6] STATUS_TASK_COMPLETE: An action used to indicate that the desired task has been completed and resets the environment. This action should also be used if the task is already completed and there is nothing more to do. For example, the task is to turn on the Wi-Fi when it is already on.
         - example: "action_type": "STATUS_TASK_COMPLETE"
      [7] STATUS_TASK_IMPOSSIBLE: An action used to indicate that the desired task is impossible to complete and resets the environment. This can result from various reasons including UI changes, Android version differences, etc.
         - example: "action_type": "STATUS_TASK_IMPOSSIBLE"
      [8] SCROLL_DOWN: Scroll down.
         - example: "action_type": "SCROLL_DOWN"
      [9] SCROLL_UP: Scroll up.
         - example: "action_type": "SCROLL_UP"
      [10] SCROLL_LEFT: Scroll left.
         - example: "action_type": "SCROLL_LEFT"
      [11] SCROLL_RIGHT: Scroll right.
         - example: "action_type": "SCROLL_RIGHT"
   (2) Corresponding screenshot before each operation. If the operation is of the "DUAL_POINT" type, the click position is marked with a red dot in the image.    
3. The positive current action and the corresponding screenshot.

## 生成负样本的准则:
在输入中给出是positive的current action,它符合下面的Level 2的标准,为了做data argumentation,我们需要生成其对应的negative current action,即下面的level 1所描述的action
   Level 1: The action is not the optimal choice for completing the task at this moment, which may lead to deviations from the task flow. For example:
      (1) Incorrect text input.
      (2) Clicking a button that might lead to an advertisement.
      (3) Announcing the task's success when it has not actually been achieved.
   Level 2: The action is the optimal and correct choice for completing the task at this moment. For example:
      (1) When showing task completion, the displayed content can fully achieve it.
      (2) When entering an unrelated interface, you can return to the main screen by executing "PRESS_HOME."
      (3) Selecting the most correct entry point to complete the current task.

## Output requirements:
- Format: {"action_desc": int, "explanation": str}. Do not include any additional characters beyond this format
- The "rating" field should be represented by the number 1 or 2 indicating the evaluation level. The "explanation" field should explain the evaluation process that led to this rating, without including descriptions of operations after the current step (future operations are considered unknown).

## Example Input:
Task Requirements: What is the capital of England?
Action and ScreenShot:
step 0: "action_type": "DUAL_POINT", "touch_point": "[0.524, 0.06]", "lift_point": "[0.524, 0.06]", "typed_text": ""
step 1: "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "capital of England"
Origin Action:
step 2: "action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""

## Example Output 1:
{
   "explanation": "Since text about the capital of England has already been entered in the search box, pressing enter directly at this step should give the answer. However, if I generate an action indicating task completion, it will seriously deviate from the current task.",
   "action_type": "STATUS_TASK_COMPLETE", 
   "touch_point": "[-1.0, -1.0]", 
   "lift_point": "[-1.0, -1.0]", 
   "typed_text": ""
}

## Example Output 2:
{
   "explanation": "Since text about the capital of England has already been entered in the search box, pressing enter directly at this step should give the answer. However, if I generate a click on the adjacent advertising area, it will deviate from the task.",
   "action_type": "DUAL_POINT", 
   "touch_point": "[0.87, 0.52]", 
   "lift_point": "[0.87, 0.52]", 
   "typed_text": ""
}

"""


prompt_negative_user = """Task Requirements: {}
Previous Action: {}
Current Action and Screenshot: {}
"""