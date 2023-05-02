import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import os
import matplotlib.patches as patches
from PIL import Image

def creatWhiteImg():
    img = np.ones((600, 600, 3), dtype = np.uint8)
    img = 255* img
    cv2.imwrite("/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/test.jpg",img)

def creatWhiteImageWithBox(img_size,box_num, fold):
    img = np.ones((img_size, img_size, 3), dtype = np.uint8)
    img = 255* img
    box_size = img_size // box_num
    for i in range(100):
        img_name = str(i) + "test.jpg"
        imagePath = os.path.join(fold,img_name)
        cv2.imwrite(imagePath,img)

        img_h = img_size
        img_w = img_size
        points = []
        for j in range(box_num):
            for k in range(box_num):
                top_left = [box_size*k,box_size*j]
                down_right = [box_size*k + box_size, box_size*j + box_size]
                box = [top_left, down_right]
                points.append(box)
        shapes = [
            dict(
                label = str(random.choice([0,1])),
                points = box,
                shape_type = "rectangle"
            )
            for box in points
        ]

        json_name = str(i) + "test.json"
        jsonPath = os.path.join(fold,json_name)
        data = dict(
            version = "5.0.2",
            flags = {},
            shapes = shapes,
            imagePath = img_name,
            imageData = None,
            imageHeight = img_h,
            imageWidth = img_w

        )

        with open(jsonPath, 'w') as f:
            json.dump(data, f)


def randomDrawBox():
    img = cv2.imread("/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yoloTest/0test.jpg")
    colors = [(0,0,255),(0,255,0)]
    for i in range(200):
        x = random.randint(0,550)
        y = random.randint(0,550)
        w = random.randint(1,50)
        h = random.randint(1,50)
        top_left =  (x,y)
        bot_right = (x+w,y+h)
        color = colors[random.choice([0,1])]
        cv2.rectangle(img, top_left, bot_right, color)
    cv2.imwrite("/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yoloTest/0test_rec.jpg",img)






if __name__ == '__main__':
    # img_size = 600
    # box_num = 20
    # fold = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yoloTest/test"
    # creatWhiteImageWithBox(img_size,box_num,fold)
    randomDrawBox()







