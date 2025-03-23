import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import os
from shutil import copy2
import pandas as pd
import argparse
from collections import Counter


def drawBoxInRawImage(fd):
    files = os.listdir(fd)
    for f in files:
        if f.endswith('json') and not f.startswith('.'):
            img = f.split(".")[0] + ".jpg"
            out_img = f.split(".")[0] + "_boxes.jpg"
            img_url = os.path.join(fd,img)
            f_url = os.path.join(fd, f)
            out_img = os.path.join(fd,out_img)

            in_img = cv2.imread(img_url)
            print(f)
            print(img)
            print(out_img)
            with open(f_url, "r") as f:
                data = json.load(f)
                for shape in data["shapes"]:
                    box_label = shape['label']
                    p1 = (int(shape['points'][0][0]),int(shape['points'][0][1]))
                    print(p1)
                    p2 = (int(shape['points'][1][0]),int(shape['points'][1][1]))
                    thickness = 8
                    if box_label == 'k':
                        color = (0,255,0)
                    elif box_label == 'c':
                        color = (0,0,255)
                    else:
                        pass
                    in_img = cv2.rectangle(in_img,p1,p2,color,thickness)
            cv2.imwrite(out_img,in_img)


    return

if __name__ == '__main__':
    # fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/ladder/source/data/rice/rescan/moreVar"
    fd = "/Volumes/work_Joe/archive/2024/WSU/ladder/app/Rice/result/train7_3_stage3/moreVar"
    drawBoxInRawImage(fd)