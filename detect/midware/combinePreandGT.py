import json
import os

def combinePreandGt(pre_json,gt_json):
    with open(gt_json, "r") as f:
        gt = json.load(f)
    for id,shape in enumerate(gt["shapes"]):
        if id in [31,32,48,55,65,66,67,77]:
            shape['label'] = 'gt_miss'
        else:
            shape['label'] = 'gt'

    with open(pre_json, "r") as f:
        pre = json.load(f)
    for shape in pre["shapes"]:
        shape['label'] = 'pre'
        gt["shapes"].append(shape)

    fd = os.path.dirname(pre_json)
    name = os.path.basename(pre_json)
    out = name.replace(".json","_combine.json")
    out = os.path.join(fd,out)
    with open(out, 'w') as f:
        json.dump(gt, f)


def preFileForAP(pre_json,gt_json):
    with open(gt_json, "r") as f:
        gt = json.load(f)
    gt_out = gt_json.replace(".json",".txt")
    with open(gt_out,'w') as f:
        for shape in gt["shapes"]:
            class_name = shape['label']
            left = shape["points"][0][0]
            top = shape["points"][0][1]
            right = shape["points"][1][0]
            bot = shape["points"][1][1]
            f.write(f"{class_name} {left} {top} {right} {bot}\n")

    with open(pre_json, "r") as f:
        pre = json.load(f)
    pre_out = pre_json.replace(".json",".txt")
    with open(pre_out,'w') as f:
        for shape in pre["shapes"]:
            class_name = shape['label']
            score = shape['score']
            left = shape["points"][0][0]
            top = shape["points"][0][1]
            right = shape["points"][1][0]
            bot = shape["points"][1][1]
            f.write(f"{class_name} {score} {left} {top} {right} {bot}\n")

    return


if __name__ == '__main__':
    pre_json = '/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/metrics/prediction/DJI_00095.json'
    gt_json = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/metrics/ground_truth/DJI_00095.json"
    # combinePreandGt(pre_json,gt_json)
    preFileForAP(pre_json,gt_json)