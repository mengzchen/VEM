import json
from typing import List
import click
import openpyxl
from openpyxl.drawing.image import Image
import cv2
from collections import defaultdict


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
        ws.cell(row=idx, column=3, value=f"step {ann["step_id"]}: {ann["action"]}")
        ws.cell(row=idx, column=4, value=ann["response"])
        ws.cell(row=idx, column=5, value=ann["rating"])

        # 插入图像
        img = Image(ann["image_path"])
        img.width, img.height = (240, 480)
        ws.row_dimensions[idx].height = 400
        ws.add_image(img, f'A{idx}')

    ws.column_dimensions['A'].width = 20
    wb.save(wpath)

def parse_rating(anns):
    statistics = defaultdict(int)
    for ann in anns:
        try:
            response = ann["response"].replace("*", "")
            rating = int(response.split("Rating: ")[1].split("\n")[0])
            ann["rating"] = rating
            statistics[rating] += 1
        except:
            ann["rating"] = -1

    colorful_print(statistics, "green")

    return anns

anns = read_json("data/aitw_train_1014_label_v4.json")
write_to_excel(anns, "v4.xlsx")