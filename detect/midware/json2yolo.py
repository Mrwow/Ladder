import json
from shutil import copy2
import os
import argparse

def yoloToJson(file_path, img_w, img_h):
    with open(file_path,'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.replace("\n","")
            l = l.split(" ")
            label = l[0]
            x1, y1 = float(l[1]), float(l[2])
            x1, x2 = x1*img_w, x2*img_w
            x2, y2 = float(l[3]), float(l[4])
            y1, y2 = y1*img_h, y2*img_h
            points = [
                [x1,y1],
                [x2,y2]
            ]
    return

def jsonToYolo(input_path):
    label_list = []
    for f in os.listdir(input_path):
        if f.endswith("json"):
            filename = os.path.join(input_path,f)
            print(filename)
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                imagePath = data["imagePath"]
                img_h = data["imageHeight"]
                img_w = data["imageWidth"]
                shapes = [
                    dict(
                        label=s["label"],
                        points=s["points"]
                    )
                    for s in data["shapes"]
                ]
            except Exception as e:
                pass
            # copy images into train data folder
            image_output_path = os.path.join(input_path,"train/images")
            imagePath = os.path.join(input_path,imagePath)
            if not os.path.exists(image_output_path):
                os.makedirs(image_output_path)
            copy2(imagePath,image_output_path)

            # label files folder
            base_name = os.path.basename(filename)
            label_name = base_name.replace('json','txt')
            labels_output_path = os.path.join(input_path,"train/labels/")
            if not os.path.exists(labels_output_path):
                os.makedirs(labels_output_path)
            label_name = os.path.join(labels_output_path,label_name)

            with open(label_name,'w') as f:
                for s in shapes:
                    label_list.append(s["label"]) if s["label"] not in label_list else label_list
                    print(label_list)
                    label_index = label_list.index(s["label"])
                    x1,y1, x2, y2=s["points"][0][0],s["points"][0][1],s["points"][1][0],s["points"][1][1]
                    w = (x2-x1)/img_w
                    h = (y2-y1)/img_h
                    x = (x1 + x2)/(2*img_w)
                    y = (y1 + y2)/(2*img_h)
                    f.write(f'{label_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
    # label summary file
    label_summary = os.path.join(input_path,"labels_summary.txt")
    with open(label_summary,"w") as f:
        for item in label_list:
            f.write(f'{item}\n')

    data_dict = dict(
        data = os.path.join(input_path,'train'),
        names = label_list
    )
    return data_dict


def jsonToYolo2(input_path):
    label_list = []
    for f in os.listdir(input_path):
        if f.endswith("json"):
            filename = os.path.join(input_path,f)
            print(filename)
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                imagePath = data["imagePath"]
                img_h = data["imageHeight"]
                img_w = data["imageWidth"]
                shapes = [
                    dict(
                        label=s["label"],
                        points=s["points"]
                    )
                    for s in data["shapes"]
                ]
            except Exception as e:
                pass
            # copy images into train data folder
            image_output_path = os.path.join(input_path,"yolo/images")
            imagePath = os.path.join(input_path,imagePath)
            if not os.path.exists(image_output_path):
                os.makedirs(image_output_path)
            copy2(imagePath,image_output_path)

            # label files folder
            base_name = os.path.basename(filename)
            label_name = base_name.replace('json','txt')
            labels_output_path = os.path.join(input_path,"yolo/labels/")
            if not os.path.exists(labels_output_path):
                os.makedirs(labels_output_path)
            label_name = os.path.join(labels_output_path,label_name)

            with open(label_name,'w') as f:
                for s in shapes:
                    label_list.append(s["label"]) if s["label"] not in label_list else label_list
                    print(label_list)
                    # label_index = label_list.index(s["label"])
                    # label = scoreToCls_4(s["label"])
                    # label = s["label"]
                    label = string2float(s["label"])
                    print(f'{s["label"]} into class: {label}')

                    x1,y1, x2, y2=s["points"][0][0],s["points"][0][1],s["points"][1][0],s["points"][1][1]
                    w = (x2-x1)/img_w
                    h = (y2-y1)/img_h
                    x = (x1 + x2)/(2*img_w)
                    y = (y1 + y2)/(2*img_h)
                    f.write(f'{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
    # label summary file
    label_summary = os.path.join(input_path,"labels_summary_2.txt")
    with open(label_summary,"w") as f:
        for item in label_list:
            f.write(f'{item}\n')

    data_dict = dict(
        data = os.path.join(input_path,'train'),
        names = label_list
    )
    return data_dict

def scoreToCls_4(score):
    score = int(score)
    if score >= 0 and score <= 20:
        score_cls = 0
    elif score > 20 and score <= 60:
        score_cls = 1
    elif score > 60 and score <= 100:
        score_cls = 2
    else:
        pass
    return score_cls

def scoreToCls(score):
    score = int(score)
    if score  <= 5:
        score_cls = 0
    elif score > 5 and score <= 20:
        score_cls = 1
    elif score > 20 and score <= 40:
        score_cls = 2
    elif score > 40 and score <= 60:
        score_cls = 3
    elif score > 60 and score <= 80:
        score_cls = 4
    elif score > 80 and score <= 100:
        score_cls = 5
    else:
        print(f'+++++++{score}')

    return score_cls

def string2float(score):
    score_cls = 2
    if score == 'st':
        score_cls = 0
    elif score == 'fr':
        score_cls = 1
    else:
        print(f"++++find other label {score}++")

    print(f"++++{score} has been changed to {score_cls}+++++")
    return score_cls



if __name__ == '__main__':
    # file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/ob/ladder_main/ladder/detect/runs/detect/exp/labels/bus.txt"
    # yoloToJson(file)

    # dir ="/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/spillman_nursery_win/aug/afterMoreBoundaryBoxes/f2/aug_centercrop"
    dir ="/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yoloTest/test"
    dir = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/ob/ladder/ladder/data/test/aug"

    jsonToYolo2(dir)


