from PIL import Image
import utils
import enum
from eval_tools.aitw import str_2_format


action_type_dict = {
    "type": "TYPE",
    "click": "DUAL_POINT",
    "press back": "PRESS_BACK",
    "press home": "PRESS_HOME",
    "press enter": "PRESS_ENTER",
    "status task complete": "STATUS_TASK_COMPLETE",
    "status task impossible": "STATUS_TASK_IMPOSSIBLE",

    "scroll down": "DUAL_POINT",
    "scroll up": "DUAL_POINT",
    "scroll left": "DUAL_POINT",
    "scroll right": "DUAL_POINT",
}

def aitw_step_update(ann):
    if ann["action_type_text"] == "scroll down":
        ann["touch"], ann["lift"] = [0.2, 0.5], [0.8, 0.5]
    elif ann["action_type_text"] == "scroll up":
        ann["touch"], ann["lift"] = [0.8, 0.5], [0.2, 0.5]
    elif ann["action_type_text"] == "scroll left":
        ann["touch"], ann["lift"] = [0.5, 0.8], [0.5, 0.2]
    elif ann["action_type_text"] == "scroll right":
        ann["touch"], ann["lift"] = [0.5, 0.2], [0.5, 0.8]
    else:
        pass

    return ann


def action2step(ann, dataset_name, image_rpath, add_visual=False):
    if dataset_name == "aitw":
        action_type = ann["action_type_id"]
        action_type_text = ann["action_type_text"]

        if action_type == 4:
            if action_type_text == "click":  
                touch_point, lift_point = ann["touch"], ann["lift"]
                click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]

                if add_visual:
                    image_rpath = utils.add_visilize2screenshot(
                        image_rpath=image_rpath,
                        action_type="click",
                        action_params=click_point
                    )

                click_point = [f"{item:.2f}" for item in click_point]
                click_point = "({},{})".format(click_point[0], click_point[1])
                action = "action_type is {}, click_point is {}.".format(action_type_text, click_point)
            else: 
                action = "action_type is {}.".format(action_type_text)
        elif action_type == 3:
            action = "action_type is {}, typed_text is {}.".format(action_type_text, ann["type_text"])
        else:
            action = "action_type is {}.".format(action_type_text)

        return action, image_rpath
    else:
        print(f"not action2step for {dataset_name}")


def convert_policy_output_to_critic_input(output_text, image_rpath):
    actor_output = str_2_format(output_text)

    if actor_output["action_type"] == "DUAL_POINT":
        touch_point = actor_output["touch_point"]
        lift_point = actor_output["lift_point"]
        click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]

        image_rpath = utils.add_visilize2screenshot(
            image_rpath=image_rpath,
            action_type="click",
            action_params=click_point
        )

        click_point = [f"{item:.2f}" for item in click_point]
        click_point = "({},{})".format(click_point[0], click_point[1])
        action = f"action_type is click, click_point is {click_point}."
    elif actor_output["action_type"] == "TYPE":
        action = f"action_type is type, typed_text is {actor_output['typed_text']}."
    else:
        # TODO the press enter is not in critic model
        action = "action_type is {}.".format(actor_output["action_type"].replace("_", " ").lower())

    return action, image_rpath


def update_trajectory(trajectories, results):
    for i in range(len(results)):
        action, image_rpath = convert_policy_output_to_critic_input(
            output_text=results[i]["output"],
            image_rpath=trajectories[i]["image_path"]
        )
        trajectories[i]["critic_input"] += action
        trajectories[i]["image_path"] = image_rpath
        trajectories[i]["next_action"] = results[i]["output"]

    return trajectories