import json
from typing import List
import click
import openpyxl
from openpyxl.drawing.image import Image
import cv2
from collections import defaultdict
import pandas as pd
import random


def read_json(rpath: str):
    with open(rpath, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data


def write_json(anns: List, wpath: str):
    json.dump(anns, open(wpath, "w"))


def add_visilize2screenshot(image_rpath, action_type, action_params):
    if action_type == "click":
        image = cv2.imread(image_rpath)
        height, width, _ = image.shape

        x = int(action_params[0] * width)
        y = int(action_params[1] * height)

        cv2.circle(image, (x, y), 20, (0, 0, 255), -1)

        image_wpath = image_rpath.split(".")[0] + "_modify.jpg"
        cv2.imwrite(image_wpath, image) 
        return image_wpath
    else:
        pass
    

def colorful_print(string: str, *args, **kwargs) -> None:
    print(click.style(string, *args, **kwargs))


def write_to_excel(anns, wpath):
    wb = openpyxl.Workbook()
    ws = wb.active

    ws.cell(row=1, column=1, value="Image")
    ws.cell(row=1, column=2, value="Task")
    ws.cell(row=1, column=3, value="Action")
    ws.cell(row=1, column=4, value="Response")
    ws.cell(row=1, column=5, value="Rating")

    for idx, ann in enumerate(anns, start=2):
        ws.cell(row=idx, column=2, value=ann["task"])
        ws.cell(row=idx, column=3, value=ann["action"])
        ws.cell(row=idx, column=4, value=ann["response"])
        ws.cell(row=idx, column=5, value=ann["rating"])

        img = Image(ann["image_path"])
        img.width, img.height = (240, 480)
        ws.row_dimensions[idx].height = 400
        ws.add_image(img, f'A{idx}')

    ws.column_dimensions['A'].width = 20
    wb.save(wpath)


def parse_rating(ann):
    try:
        response = ann["response"].replace("*", "")
        rating = int(response.split("Rating: ")[1].split("\n")[0])
        return rating
    except:
        return -1
    

def write_jsonl(anns, wpath):
    with open(wpath, 'w', encoding='utf - 8') as f:
        for item in anns:
            json_line = json.dumps(item)
            f.write(json_line + '\n')


def read_jsonl(rpath):
    data = []
    with open(rpath, 'r', encoding='utf - 8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except:
                colorful_print(f"Error decoding JSON on line: {idx}", "red")
    return data


def read_xlsx(rpath):
    data = pd.read_excel(rpath)
    return data.to_dict(orient="records")


def check_fail_case():
    anns = read_xlsx("data/aitz_failure_index.xlsx")
    aitw_anns = read_json("data/aitw_data_train.json")

    # general single webshopping install googleapps
    aitw_anns = aitw_anns["install"] + aitw_anns["general"] + aitw_anns["single"] + aitw_anns["webshopping"] + aitw_anns["googleapps"]

    aitw_dict = defaultdict()
    for aitw_ann in aitw_anns:
        last_case = aitw_ann[-1]
        aitw_dict[last_case["ep_id"]] = last_case

    count = 0
    wb = openpyxl.Workbook()
    ws = wb.active

    ws.cell(row=1, column=1, value="Image")
    ws.cell(row=1, column=2, value="Instruction")
    ws.cell(row=1, column=2, value="Task")

    for idx, ann in enumerate(anns, start=2):
        try:
            info = aitw_dict[ann["ep_id"]]
            ws.cell(row=idx, column=2, value=ann["instruction"])
            ws.cell(row=idx, column=3, value=info["goal"])
            image_path = f"data/images/aitw_images/{info["img_filename"]}.png"
            img = Image(image_path)
            img.width, img.height = (240, 480)
            ws.row_dimensions[idx].height = 400
            ws.add_image(img, f'A{idx}')
            count += 1
        except:
            print(ann["ep_id"])

    ws.column_dimensions['A'].width = 20
    wb.save("fail.xlsx")


def sample_data(rpath, wpath):
    sample_anns = []
    statistics = defaultdict(int)
    last_step_statistics = defaultdict(int)
    anns = read_jsonl(rpath)
    for ann in anns:
        statistics[ann["rating"]] += 1
        if ann["step_id"] == len(ann["action_list"]) - 1:
            last_step_statistics[ann["rating"]] += 1
            sample_anns.append(ann)

    colorful_print(statistics, "green")
    colorful_print(last_step_statistics, "green")

    write_to_excel(sample_anns, wpath)
