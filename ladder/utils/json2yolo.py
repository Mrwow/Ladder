import json
from shutil import copy2
import os
import random
import shutil
import glob

import cv2
import yaml

from ladder.widgets.label_file import LabelFile
from .imgAug import grid2tileBatch

# def jsonToYolo_bk(input_path):
#     label_list = []
#     for f in os.listdir(input_path):
#         if f.endswith("json"):
#             filename = os.path.join(input_path,f)
#             print(filename)
#             try:
#                 with open(filename, "r") as f:
#                     data = json.load(f)
#                 imagePath = data["imagePath"]
#                 img_h = data["imageHeight"]
#                 img_w = data["imageWidth"]
#                 shapes = [
#                     dict(
#                         label=s["label"],
#                         points=s["points"]
#                     )
#                     for s in data["shapes"]
#                 ]
#             except Exception as e:
#                 pass
#             # copy images into train data folder
#             image_output_path = os.path.join(input_path,"train/images")
#             if not os.path.exists(image_output_path):
#                 os.makedirs(image_output_path)
#             imagePath = os.path.join(input_path,imagePath)
#             copy2(imagePath,image_output_path)

#             # label files folder
#             base_name = os.path.basename(filename)
#             label_name = base_name.replace('json','txt')
#             labels_output_path = os.path.join(input_path,"train/labels/")
#             if not os.path.exists(labels_output_path):
#                 os.makedirs(labels_output_path)
#             label_name = os.path.join(labels_output_path,label_name)

#             with open(label_name,'w') as f:
#                 for s in shapes:
#                     label_list.append(s["label"]) if s["label"] not in label_list else label_list
#                     print(label_list)
#                     label_index = label_list.index(s["label"])
#                     x1,y1, x2, y2=s["points"][0][0],s["points"][0][1],s["points"][1][0],s["points"][1][1]
#                     w = (x2-x1)/img_w
#                     h = (y2-y1)/img_h
#                     x = (x1 + x2)/(2*img_w)
#                     y = (y1 + y2)/(2*img_h)
#                     f.write(f'{label_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

#     # label summary file
#     label_summary = os.path.join(input_path,"labels_summary.txt")
#     with open(label_summary,"w") as f:
#         for item in label_list:
#             f.write(f'{item}\n')

#     # generate train_data.yaml
#     names = {}
#     for i,item in enumerate(label_list):
#         names[i] = item

#     data_yaml = os.path.join(input_path,"train_data.yaml")
#     train_data_config = {
#         'path': input_path,
#         'train': "train",
#         'val': 'train',
#         "names": names
#     }
#     with open(data_yaml, 'w', encoding="utf-8") as f:
#         yaml.dump(train_data_config,f)

#     data_dict = dict(
#         data = os.path.join(input_path,'train'),
#         names = label_list
#     )
#     return data_dict


def jsonToYolo(input_path):
    image_labeled = len(glob.glob(os.path.join(input_path, "*.json")))
    print(image_labeled)
    if image_labeled <=10:
        jsonToYoloSameTrainTest(input_path=input_path)
    else:
        jsonToYoloTrainTestSplit(input_path=input_path)


def jsonToYoloSameTrainTest(input_path):
    label_list = []

    image_output_path = os.path.join(input_path,"train/images")
    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)
    labels_output_path = os.path.join(input_path,"train/labels/")
    if not os.path.exists(labels_output_path):
        os.makedirs(labels_output_path)

    for f in os.listdir(input_path):
        if f.endswith("jpg") or f.endswith("JPG") or f.endswith("png"):
            # copy image
            imagePath = os.path.join(input_path,f)

            imagePath = os.path.join(input_path,imagePath)
            copy2(imagePath,image_output_path)
            print(f"img is {f}")

            # create txt file for yolo
            img = os.path.basename(f)
            img_json = img.split(".")[0] + ".json"
            img_json_url = os.path.join(input_path,img_json)

            if os.path.exists(img_json_url):
                print(f"json is {img_json}")
                try:
                    with open(img_json_url, "r") as f:
                        data = json.load(f)
                    # imagePath = data["imagePath"]
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

                # label files folder
                label_name = img_json.replace('json','txt')

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
            else:
                print("create a empty txt file")
                # label files folder
                label_name = img_json.replace('json','txt')
                labels_output_path = os.path.join(input_path,"train/labels/")
                if not os.path.exists(labels_output_path):
                    os.makedirs(labels_output_path)
                label_name = os.path.join(labels_output_path,label_name)
                with open(label_name,'w') as f:
                    pass

    # label summary file
    label_summary = os.path.join(input_path,"train","labels_summary.txt")
    with open(label_summary,"w") as f:
        for item in label_list:
            f.write(f'{item}\n')

    # generate train_data.yaml
    names = {}
    for i,item in enumerate(label_list):
        names[i] = item

    data_yaml = os.path.join(input_path,"train","train_data.yaml")
    train_data_config = {
        'path': input_path,
        'train': "train",
        'val': 'train',
        "names": names
    }
    with open(data_yaml, 'w', encoding="utf-8") as f:
        yaml.dump(train_data_config,f)

    data_dict = dict(
        data = os.path.join(input_path,'train'),
        names = label_list
    )
    return data_dict


def jsonToYoloTrainTestSplit(input_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    # jsonToYoloSplitFold
    # create folder for "train", "val", "test"
    fold_list = ["train", "val", "test"]
    base_fd = os.path.dirname(input_path)
    for fd in fold_list:
        # fd_img = os.path.join(base_fd,"yolo",fd)
        fd_img = os.path.join(input_path,"train_test", fd)
        if not os.path.exists(fd_img):
            os.makedirs(fd_img)
        # fd_json = os.path.join(base_fd,"yolo",fd)
        fd_json = os.path.join(input_path,"train_test",fd)
        if not os.path.exists(fd_json):
            os.makedirs(fd_json)

    # get all image name, json file name, and label list
    list_image = []
    list_label = []
    list_json = []
    for f in os.listdir(input_path):
        if not f.startswith(".") and (f.endswith("jpg") or f.endswith("JPG") or f.endswith("png")): 
            list_image.append(f)
            img = os.path.basename(f)
            img_json = img.split(".")[0] + ".json"
            img_json_url = os.path.join(input_path,img_json)
            list_json.append(img_json_url)
            if os.path.exists(img_json_url):
                try:
                    with open(img_json_url, "r") as f:
                        data = json.load(f)
                    shapes = [
                        dict(
                            label=s["label"],
                        )
                        for s in data["shapes"]
                    ]
                    for s in shapes:
                        if s["label"] not in list_label:
                            list_label.append(s["label"])
                except Exception as e:
                    pass

    # generate yolo_train_config.yaml
    sorted_list_label = list(enumerate(list_label))
    sorted_list_label = sorted(sorted_list_label, key=lambda x: x[1])
    dict_id_label = {}
    dict_label_id = {}
    for i in range(len(sorted_list_label)):
        dict_id_label[i] = sorted_list_label[i][1]
        dict_label_id[sorted_list_label[i][1]] = i

    data_yaml = os.path.join(base_fd, "yolo","yolo_train_config.yaml")
    train_data_config = {
        'path': os.path.join(base_fd,"yolo"),
        'train': "train",
        'val': 'val',
        'test':'test',
        "names": dict_id_label
    }
    with open(data_yaml, 'w', encoding="utf-8") as f:
        yaml.dump(train_data_config,f)

    # copy image and json file
    indx_img_list = list(range(len(list_image)))
    len_imgs = len(indx_img_list)
    random.seed(random_seed)
    random.shuffle(indx_img_list)
    train_stop_flag = len_imgs * train_ratio
    val_stop_flag = len_imgs * (train_ratio + val_ratio)
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in indx_img_list:
        url_img = os.path.join(input_path,list_image[i])
        url_json = os.path.join(input_path, list_json[i])
        json_file_name = os.path.basename(url_json)
        if current_idx <= train_stop_flag:
            targt_img = os.path.join(base_fd,"yolo","train")
            targt_json = os.path.join(base_fd,"yolo","train",json_file_name)
            train_num = train_num + 1
        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            targt_img = os.path.join(base_fd,"yolo","val")
            targt_json = os.path.join(base_fd,"yolo","val",json_file_name)
            val_num = val_num + 1
        else:
            targt_img = os.path.join(base_fd,"yolo","test")
            targt_json = os.path.join(base_fd,"yolo","test",json_file_name)
            test_num = test_num + 1
        
        copy2(url_img, targt_img)
        copy2(url_json,targt_json)
        current_idx = current_idx + 1

    
    for fd in fold_list:
        fd_img_folder = os.path.join(base_fd,"yolo",fd)
        sub_img_fd = os.path.join(fd_img_folder,"images")
        sub_yolo_fd = os.path.join(fd_img_folder,"lables")
        if not os.path.exists(sub_img_fd):
            os.makedirs(sub_img_fd)
        if not os.path.exists(sub_yolo_fd):
            os.makedirs(sub_yolo_fd)

        # get image patch
        grid2tileBatch(fd=fd_img_folder, grid_size=1200, min_visi=0.60)

        fd_img_grid = os.path.join(fd_img_folder,"grids")
        # create yolo txt for each image
        for f in os.listdir(fd_img_grid):
            if f.endswith('json') and not f.startswith('.'):
                tile_name = f.replace(".json",".JPG")
                tile_url = os.path.join(fd_img_grid,tile_name)
                copy2(tile_url, sub_img_fd)

                tile_json = os.path.join(fd_img_grid,f)
                txt_name = f.replace(".json",".txt")
                tile_txt = os.path.join(sub_yolo_fd, txt_name)

                try:
                    with open(tile_json, "r") as f:
                        data = json.load(f)
                    # imagePath = data["imagePath"]
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

                # yolo txt output
                print(f"conver {tile_json} to {tile_txt}")
                with open(tile_txt,'w') as f:
                    for s in shapes:
                        label_index = dict_label_id[s["label"]]
                        x1,y1, x2, y2=s["points"][0][0],s["points"][0][1],s["points"][1][0],s["points"][1][1]
                        w = (x2-x1)/img_w
                        h = (y2-y1)/img_h
                        x = (x1 + x2)/(2*img_w)
                        y = (y1 + y2)/(2*img_h)
                        f.write(f'{label_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

    data_dict = dict(
        data_url = data_yaml,
        names = list_label
    )
    
    return data_dict



def ultraResult2Json(results):
    for result in results:
        path = result.path
        work_dir = os.path.dirname(path)
        img = os.path.basename(path)
        json_out = img.split('.')[0] + ".json"
        json_url = os.path.join(work_dir,json_out)

        lf = LabelFile()
        shapes = []
        boxes = result.boxes.xyxy
        probs = result.boxes.conf
        cls = result.boxes.cls
        names_dict = result.names
        h,w = result.orig_shape
        for i, box in enumerate(boxes):
            shape=dict(
                label=names_dict[cls[i].item()],
                points= [[box[0].item(), box[1].item()], [box[2].item(), box[3].item()]],
                shape_type="rectangle",
                group_id=None,
                flags = {
                    "prob": probs[i].item()
                }
            )
            shapes.append(shape)
        try:
            lf.save(
                filename=json_url,
                shapes=shapes,
                imagePath=img,
                imageData=None,
                imageHeight=h,
                imageWidth=w,
                otherData=None,
                flags={},
            )
        except Exception as e:
            raise e
    # if os.path.isfile(data):
        # work_dir = os.path.dirname(data)
        # img = os.path.basename(data)
        # json_out = img.split('.')[0] + ".json"
        # json_url = os.path.join(work_dir,json_out)

