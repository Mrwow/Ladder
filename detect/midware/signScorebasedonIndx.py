import os
import json
import pandas as pd
import shutil


def signScoreBasedOnIndx(json_file,indx_file):
    dir_name = os.path.dirname(json_file)
    file_name = os.path.basename(json_file)
    img_name = file_name.replace('.json','.JPG')
    img_in_src = os.path.join(dir_name,img_name)

    dir_out = os.path.join(dir_name,'score')
    json_out = os.path.join(dir_out,file_name)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    shutil.copy(img_in_src,dir_out)

    with open(json_file, "r") as f:
        data = json.load(f)

    indx_score = pd.read_csv(indx_file)
    h = data["imageHeight"]
    w = data["imageWidth"]

    # print(indx_score.head())

    for shape in data["shapes"]:
        indx = shape["label"]
        points = shape["points"]

        points_checked = checkPoints(points, w, h)
        score = indx_score.loc[indx_score['col_row'] == indx, 'Score'].item()
        if not indx.find("_"):
            print(f'{indx} score is {score}')
        shape["label"] = str(score)
        shape["points"] = points_checked
        if score == -100:
            print(file_name)
            print(f'{indx} score is {score}')

    with open(json_out, 'w') as f:
        json.dump(data, f)

def checkPoints(points,img_w, img_h):
    new_points = []
    for p in points:
        x = p[0]
        y = p[1]
        if x < 0:
            x = 0
        elif x > img_w:
            x = img_w
        else:
            x = x
        if y < 0:
            y = 0
        elif y > img_h:
            y = img_h
        else:
            y = y
        new_points.append([x,y])
    x1,y1 = new_points[0][0],new_points[0][1]
    x2,y2 = new_points[1][0],new_points[1][1]
    new_points_out = []
    if x1 > x2:
        top_left = [x2,y2]
        bot_right = [x1,y1]
    else:
        top_left = [x1,y1]
        bot_right = [x2,y2]
    new_points_out = [top_left,bot_right]
    return new_points_out

def batch_signScoreBasedOnIndx(fd,indx_file):
    for f in os.listdir(fd):
        if f.endswith('.json'):
            # print(f)
            json_file = os.path.join(fd,f)
            signScoreBasedOnIndx(json_file=json_file,indx_file=indx_file)


if __name__ == '__main__':
    # json_file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/signscore/DJI_00001.json"
    indx_file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/signscore/idxScoreIT.csv"
    # signScoreBasedOnIndx(json_file=json_file,indx_file=indx_file)
    fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/signscore/"
    batch_signScoreBasedOnIndx(fd=fd, indx_file=indx_file)