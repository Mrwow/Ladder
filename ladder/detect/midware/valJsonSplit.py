import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

def yoloValJsonSplit2Json(json_in, img_fd):
    """
    input:
    - json_in: json file from val.py

    output:
    -
    """
    with open(json_in, "r") as f:
        data = json.load(f)

    img_list = []
    for box in data:
        img_list.append(box['image_id'])
    img_list = list(set(img_list))
    print(img_list)

    for img in img_list:
        shapes = []
        img_name = img + '.JPG'
        img_url = os.path.join(img_fd,img_name)
        if os.path.exists(img_url):
            im = Image.open(img_url)
            im_w, im_h = im.size
            print(img_name)
            for box in data:
                if box['image_id'] == img:
                    top_left = [box['bbox'][0],box['bbox'][1]]
                    bot_right = [box['bbox'][0]+box['bbox'][2], box['bbox'][1]+box['bbox'][3]]
                    bbox = dict(
                        points = [top_left,bot_right],
                        label = str(box["category_id"]),
                        score = box["score"],
                        shape_type = "rectangle",
                        flags = {},
                        group_id =None,
                        other_data = {},
                    )
                    shapes.append(bbox)
            print(img)
            print(f"loading {len(shapes)} shape")

            data_out = dict(
                version="5.0.2",
                flags={},
                shapes=shapes,
                imagePath=img_url,
                imageData=None,
                imageHeight=im_h,
                imageWidth=im_w,
            )
            json_out = img + ".json"
            json_out_url = os.path.join(img_fd, json_out)
            with open(json_out_url, "w") as f:
                json.dump(data_out, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    json_in = "/Users/ZhouTang/Downloads/2023/2023_intern/source/ladder/ladder/data/test/aug/yolo/ccrop/exp44/best_predictions.json"
    img_fd = "/Users/ZhouTang/Downloads/2023/2023_intern/source/ladder/ladder/data/test/aug/yolo/ccrop/images/"
    yoloValJsonSplit2Json(json_in, img_fd)