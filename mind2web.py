import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image 
import json
import os
from tqdm import tqdm
import random
import argparse

from utils import read_json


# show image with bbox
def show_image_with_bbox(image, bbox=None):

    img_width, img_height = image.size
    dpi = 40
    figsize = img_width / float(dpi), img_height / float(dpi)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bbox:
        x = int(bbox['x'])
        y = int(bbox['y'])
        width = int(bbox['width'])
        height = int(bbox['height'])
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()


# convert action to prediction format
def action2step(action, image_size):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']  # five types of data

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{item:.2f}" for item in click_point]
    click_point = "({},{})".format(click_point[0], click_point[1])

    if action_type in ['CLICK', 'HOVER', 'ENTER']:  # following mind2web, these three actions are regarded as click
        action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(2, click_point, select_value)
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(3, click_point, typed_text)
    return action_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', type=str, default="data/images/mind2web_images")
    parser.add_argument('--anns_rpath', type=str, default="data/mind2web_data_train.json")
    parser.add_argument('--anns_wpath', type=str, default="data/mind2web_train_sft.json")
    args = parser.parse_args()

    mind2web_imgs_dir = args.imgs_dir
    mind2web_train = read_json(args.anns_rpath)
    print(f"### orgin data size: {len(mind2web_train)}")

    train_step = []
    prompt_origin = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
    step_i = 0
    for episode in tqdm(mind2web_train):
        # action_reprs: [link]  NBA -> HOVER', '[link]  Scores -> CLICK', '[span]  Mon -> CLICK']
        goal = episode["confirmed_task"]
        annot_id = episode["annotation_id"]
        previous_actions = []

        for step in episode["actions"]:
            # Few actions can not find its corresponding bbox, jump these actions
            if "bbox" not in step:
                print("action not found")
                continue

            filename = annot_id + '-' + step["action_uid"] + '.jpg'
            img_path = os.path.join(mind2web_imgs_dir, filename)
            if not os.path.exists(img_path):
                print("img not found")
                input()
            image = Image.open(img_path)
            
            previous_step = ""
            for i, action in enumerate(previous_actions[-4:]):
                previous_step += 'Step' + str(i) + ': ' + action + ". "
            
            # "action_type": 4, "click_point": (0.16,0.12)
            action_step = action2step(step, image.size)
            previous_actions.append(action_step)

            prompt = prompt_origin.format(goal, previous_step)

            conversations = []
            conv_user = {"role": "user", "content": "Picture 1: <image>\n"}
            conv_user["content"] += prompt
            conv_ai = {"role": "assistant", "content": str(action_step)}
            conversations.append(conv_user)
            conversations.append(conv_ai)

            train_step.append({"messages": conversations, "images": [img_path]})
            step_i += 1

            # visualize step data
            # show_image_with_bbox(image, step["bbox"])
            # print(conversations)
            # input()

    random.shuffle(train_step)
    print("Num of total step: " + str(len(train_step)))
    json.dump(train_step, open(args.anns_wpath, "w"))
    print(train_step[0])
