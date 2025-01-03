import utils
from eval_tools.aitw import str_2_format
from data_preprocess.prompt import prompt_critic_system, prompt_critic_user
from eval_tools.aitw import get_roll_type


action_type_dict = {
    "type": "TYPE",
    "click": "DUAL_POINT",
    "press back": "PRESS_BACK",
    "press home": "PRESS_HOME",
    "press enter": "PRESS_ENTER",
    "status task complete": "STATUS_TASK_COMPLETE",
    "status task impossible": "STATUS_TASK_IMPOSSIBLE",
    "scroll down": "SCROLL_DOWN",
    "scroll up": "SCROLL_UP",
    "scroll left": "SCROLL_LEFT",
    "scroll right": "SCROLL_RIGHT",
}


scroll_map = {
    "up": [[0.8000, 0.5000], [0.2000, 0.5000]],
    "down": [[0.2000, 0.5000], [0.8000, 0.5000]],
    "left": [[0.5000, 0.8000], [0.5000, 0.2000]],
    "right": [[0.5000, 0.2000], [0.5000, 0.8000]]
}


def extract_scroll(action):
    if action["touch_point"] == action["lift_point"]:
        return action
    
    scroll_type = get_roll_type(action["touch_point"], action["lift_point"])
    if scroll_type == "up":
        action["action_type"] = "SCROLL_UP"
    elif scroll_type == "down":
        action["action_type"] = "SCROLL_DOWN"
    elif scroll_type == "left":
        action["action_type"] = "SCROLL_LEFT"
    elif scroll_type == "right":
        action["action_type"] = "SCROLL_RIGHT"
    else:
        return action
    
    return action


def update_trajectory(anns, results):
    for (result, ann) in zip(results, anns):
        new_action = str_2_format(result["output"])
        new_action = extract_scroll(new_action)
        new_action_desc = step_2_action(
            new_action["action_type"],
            new_action["touch_point"],
            new_action["lift_point"],
            new_action["typed_text"],
            add_all_dict=False
        )
        
        history_action_desc = "\n".join(ann["action_desc_list"][:ann["step_id"] - 1]) + "\n" + new_action_desc
        
        ann["critic_input"] = prompt_critic_system + prompt_critic_user.format(ann["task"], history_action_desc, new_action_desc)
        ann["policy_output"] = new_action_desc
        ann["critic_image"] = utils.add_visilize2screenshot(ann["policy_image"], new_action, "policy")

    return anns


def step_2_action(action_type, touch_point, lift_point, typed_text, add_all_dict):
    if add_all_dict:
        # for auto gui input
        if "SCROLL" in action_type:
            point_pair = scroll_map[action_type]
            return f"\"action_type\": \"DUAL_POINT\", \"touch_point\": \"{point_pair[0]}\", \"lift_point\": \"{point_pair[1]}\", \"typed_text\": \"{typed_text}\""
        else:
            return f"\"action_type\": \"{action_type}\", \"touch_point\": \"{touch_point}\", \"lift_point\": \"{lift_point}\", \"typed_text\": \"{typed_text}\""
    else:
        if action_type == "DUAL_POINT":
            return f"\"action_type\": \"{action_type}\", \"click_point\": \"{touch_point}\""
        elif action_type == "TYPE":
            return f"\"action_type\": \"{action_type}\", \"typed_text\": \"{typed_text}\""
        else:
            return f"\"action_type\": \"{action_type}\""