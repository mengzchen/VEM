from PIL import Image

import utils

def action2step(ann, dataset_name, image_rpath):
    if dataset_name == "mind2web":
        action_type = ann["operation"]["original_op"]
        assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER'], f"action: action_type"
        point_x = ann["bbox"]["x"] + (ann["bbox"]["width"] / 2)
        point_y = ann["bbox"]["y"] + (ann["bbox"]["height"] / 2)

        image_size = Image.open(image_rpath).size
        # follow Qwen2VL setting
        click_point = [(point_x / image_size[0]) * 1000, (point_y / image_size[1]) * 1000]
        click_point = [int(item) for item in click_point]

        action_example = {}
        if action_type in ['CLICK', 'HOVER', 'ENTER']: 
            action_example["function"] = "click"
            action_example["args"] = {"point": click_point}
        elif action_type == 'SELECT':
            select_value = ann["operation"]["value"]
            action_example["function"] = "select"
            action_example["args"] = {"point": click_point, "choose": select_value}
        elif action_type == 'TYPE':
            typed_text = ann["operation"]["value"]
            action_example["function"] = "select"
            action_example["args"] = {"point": click_point, "text": typed_text}

        return action_example
    elif dataset_name == "aitw":
        action_type = ann["action_type_id"]
        action_type_text = ann["action_type_text"]

        if action_type == 4:
            if action_type_text == "click":  
                touch_point = ann["touch"]
                lift_point = ann["lift"]
                click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]

                # add point in screenshot
                image_rpath = utils.add_visilize2screenshot(image_rpath=image_rpath, action_type="click", action_params=click_point)

                click_point = [f"{item:.2f}" for item in click_point]
                click_point = "({},{})".format(click_point[0], click_point[1])
                action = "action_type is {}, click_point is {}.".format(action_type_text, click_point)
            else: 
                action = "action_type is {}.".format(action_type_text)
        elif action_type == 3:
            action = "action_type is {}, typed_text is {}.".format(action_type_text, step["type_text"])
        else:
            action = "action_type is {}.".format(action_type_text)

        return action, image_rpath
    else:
        utils.colorful_print(f"not action2step for {dataset_name}", "red")

class GUIDataset:
    def __init__(self) -> None:
        pass

    