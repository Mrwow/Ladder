import albumentations as A
import cv2

import json
import os
import random


def checkBox(bboxes, w, h):
    new_boxes= []
    for i, box in enumerate(bboxes):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2, w)
        y2 = min(y2, h)
        box = [x1,y1,x2,y2,box[4]]

        new_box = []
        if x1 == x2 or y1 == y2:
            print(box)
            pass
        elif x1 < x2 and y1 < y2:
            new_box = box
        elif x1 > x2 and y1 > y2:
            print(box)
            new_box = [x2,y2,x1,y1,box[4]]
        elif x1 < x2 and y1 > y2:
            print(box)
            new_box = [x1,y2,x2,y1,box[4]]
        elif x1 > x2 and y1 < y2:
            print(box)
            new_box = [x2,y1,x1,y2,box[4]]
        new_boxes.append(new_box)
    return new_boxes

def cropJson(img_url, json_url, out_dir, pts, min_visi=0.5):
    # load image
    image = cv2.imread(img_url)
    h, w ,c =image.shape

    # load json and do bboxes check
    print(json_url)
    with open(json_url, "r") as f:
        data = json.load(f)
    bboxes = [
        [
            s["points"][0][0],
            s["points"][0][1],
            s["points"][1][0],
            s["points"][1][1],
            s["label"]
        ]
        for s in data["shapes"]
    ]
    bboxes_ck = checkBox(bboxes, w , h)

    # crop transform
    transform_crop = A.Compose([
        A.Crop(x_min=pts[0],y_min=pts[1],x_max=pts[2],y_max=pts[3])
    ], bbox_params=A.BboxParams(format='pascal_voc',min_visibility=min_visi))
    transformed = transform_crop(image=image, bboxes=bboxes_ck)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    print(image.shape)
    print(transformed_image.shape)
    h_c, w_c, c_c = transformed_image.shape
    print(f"after cropping, h_c is {h_c}, w_c is {w_c}")
    print("++++++++++++++")
    image_name = os.path.basename(img_url)
    image_out = image_name.split('.')[0]+f"_{pts[0]}_{pts[1]}_{pts[2]}_{pts[3]}."+image_name.split('.')[1]
    json_out = image_name.split('.')[0]+f"_{pts[0]}_{pts[1]}_{pts[2]}_{pts[3]}.json"
    json_out = os.path.join(out_dir,json_out)

    data["imageData"] = None
    data["imagePath"] = image_out
    data["imageHeight"] = h_c
    data["imageWidth"] = w_c
    data["shapes"] = [
        dict(
            label=box[4],
            points=[[box[0],box[1]],[box[2],box[3]]],
            shape_type= "rectangle",
            flags={},
            group_id= None
        )
        for box in transformed_bboxes
    ]
    with open(json_out,'w') as outfile:
        json.dump(data, outfile)

    return

def jsonBoxrotate(img_url,index,outdir,transform, transform_name):
    dirname = os.path.dirname(img_url)
    img_name = os.path.basename(img_url).replace(".JPG","")
    in_json = img_name + ".json"
    in_json = os.path.join(dirname,in_json)

    out_json = img_name + "_" + str(index) + "_" + transform_name + ".json"
    img_out = out_json.replace("json", "JPG")
    out_json = os.path.join(outdir,out_json)
    img_out = os.path.join(outdir,img_out)

    image = cv2.imread(img_url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w, h ,c =image.shape

    with open(in_json, "r") as f:
        data = json.load(f)
    bboxes = [
        [
            s["points"][0][0],
            s["points"][0][1],
            s["points"][1][0],
            s["points"][1][1],
            s["label"]
        ]
        for s in data["shapes"]
    ]
    bboxes = checkBox(bboxes, w , h)

    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    print("++++++++++++++")
    print(image.shape)
    print(transformed_image.shape)
    print("++++++++++++++")

    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_out,transformed_image)
    data["imageData"] = None
    data["imagePath"] = os.path.basename(img_out)
    data["imageHeight"] = transformed_image.shape[0]
    data["imageWidth"] = transformed_image.shape[1]
    data["shapes"] = [
        dict(
            label=box[4],
            points=[[box[0],box[1]],[box[2],box[3]]],
            shape_type= "rectangle",
            flags={},
            group_id= None
        )
        for box in transformed_bboxes
    ]
    with open(out_json,'w') as outfile:
        json.dump(data, outfile)

def batchRotate(dir,transform, transform_name, num=1, dig=90):
    out_dir = os.path.join(dir,"aug")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for f in os.listdir(dir):
        if f.endswith("JPG") and not f.startswith("."):
            img_url = os.path.join(dir, f)
            print(img_url)
            for i in range(num):
                if transform == 'rotate':
                    trans = rotateTransform(img_url, dg=dig)
                    jsonBoxrotate(img_url,index=i,outdir=out_dir,transform=trans, transform_name=transform_name)
                else:
                    jsonBoxrotate(img_url,index=i,outdir=out_dir,transform=transform, transform_name=transform_name)


def rotateTransform(img_url, dg=90):
    image = cv2.imread(img_url)
    w, h, c = image.shape
    pad_val = max(w,h)
    if dg == 90 or dg == -90:
        trans = A.Compose([
            A.PadIfNeeded(min_height=pad_val,min_width=pad_val,border_mode=cv2.BORDER_CONSTANT),
            A.Rotate(limit=[dg,dg],p=1),
            A.CenterCrop(height=h,width=w)
        ], bbox_params=A.BboxParams(format='pascal_voc',min_visibility=0.6))
    elif dg == 0:
        trans = A.Compose([
            A.CenterCrop(height=w,width=h)
        ], bbox_params=A.BboxParams(format='pascal_voc',min_visibility=0.6))
    else:
        pass

    return  trans

if __name__ == '__main__':
    # crop alfalfa
    random.seed(21)
    dir = "/Users/ZhouTang/Downloads/2023/2023_intern/source/ladder/ladder/data/test"
    # centercrop
    transform_ccrop = A.Compose([
        A.CenterCrop(height=2200,width=2200)
    ], bbox_params=A.BboxParams(format='pascal_voc',min_visibility=0.3))
    batchRotate(dir,transform = transform_ccrop, transform_name="centerCrop",num=1, dig=90)





