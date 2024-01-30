import json
import os
from shutil import copy2
import pandas as pd
import argparse
from collections import Counter


def checkIfImageHaveJson(fd, out_fd):
    files = os.listdir(fd)

    for f in files:
        if f.endswith(".json"):
            img = f.split(".")[0] + ".jpg"
            img_url = os.path.join(fd,img)
            f_url = os.path.join(fd, f)
            print(f)
            print(img)
            copy2(img_url,out_fd)
            copy2(f_url,out_fd)

    return

def countBox_FL(json_fd,var):
    imgs = []
    labels = []
    box_num_list = []
    for f in os.listdir(json_fd):
        if f.endswith("json"):
            json_url = os.path.join(json_fd,f)
            with open(json_url, "r") as f:
                data = json.load(f)
                imagePath = data["imagePath"]
                box_num = len(data["shapes"])
            imgs.append(imagePath)
            box_num_list.append(box_num)
            if box_num > 0:
                print(f"image {imagePath} have {box_num} box been found!")
                labels.append("FL")
            else:
                labels.append("N")
    df = {
        'Img_id': imgs,
        'Prediction':labels,
        'Findings':box_num_list
    }
    df = pd.DataFrame(df)
    out_path = os.path.join(json_fd , f'1_{var}_prediction_yolo.csv')
    df.to_csv(out_path, index=False)

def countBox(json_fd, var):
    imgs = []
    label_list = []
    box_num_list= []
    for f in os.listdir(json_fd):
        if f.endswith("json"):
            json_url = os.path.join(json_fd,f)
            with open(json_url, "r") as f:
                data = json.load(f)
                imagePath = data["imagePath"]
                box_num = len(data["shapes"])
                labels = []
                for shape in data["shapes"]:
                    labels.append(shape['label'])
                labels = dict(Counter(labels))

            imgs.append(imagePath)
            box_num_list.append(box_num)
            label_list.append(labels)
            print(f"image {imagePath} have {box_num} box, {labels}, been found!")


    df = {
        'Img_id': imgs,
        'Findings':box_num_list,
        'Prediction':label_list
    }
    df = pd.DataFrame(df)
    out_path = os.path.join(json_fd , f'1_{var}_prediction_yolo.csv')
    df.to_csv(out_path, index=False)


parser = argparse.ArgumentParser(description="Define the data folder, gpu and out name")
parser.add_argument('--data', '-d', help='the data folder', required=True)
parser.add_argument('--var', '-v', help='variety', required=True)
args = parser.parse_args()


if __name__ == '__main__':
    countBox(json_fd=args.data,var=args.var)

# if __name__ == '__main__':
#     # fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/ladder/source/data/wheet_flower/xianran/bigbluestem/N"
#     # out_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/ladder/source/data/wheet_flower/xianran/ladder_label/bigbluestem/N"
#     # checkIfImageHaveJson(fd,out_fd)
#     fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/ladder/source/data/wheet_flower/xianran/ladder_label/test/FL/littlebluestem/FL"
#     countBox(fd,var="littlebluestem")