# import imageio
# from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
# from imgaug import augmenters as iaa
import albumentations as A
import cv2

import json
from PIL import Image as im
from matplotlib import pyplot as plt
import os
import random


# def __jsonBoxrotate__(img_url, json_file,out_json, rotate=270):
#     img = imageio.imread(img_url)
#     print(img.shape)
#     with open(json_file, "r") as f:
#         data = json.load(f)
#     bbs = [
#         BoundingBox(x1 = s["points"][0][0],
#                     y1 = s["points"][0][1],
#                     x2 = s["points"][1][0],
#                     y2 = s["points"][1][1],
#                     label=s["label"])
#         for s in data["shapes"]
#     ]
#     bbs = BoundingBoxesOnImage(bbs, shape=img.shape)
#     rot = iaa.Affine(rotate=rotate)
#     image_aug, bbs_aug = rot(image=img, bounding_boxes=bbs)
#     bbs_aug_move = bbs_aug.remove_out_of_image().clip_out_of_image()
#     print(len(bbs_aug))
#     print(len(bbs_aug_move))
#
#     image_aug = im.fromarray(image_aug)
#     img_out = out_json.replace("json", "JPG")
#     image_aug.save(img_out)
#
#     data["imageData"] = None
#     data["imagePath"] = os.path.basename(img_out)
#     data["shapes"] = [
#         dict(
#             label=box.label,
#             points=[[float(box.x2),float(box.y2)],[float(box.x1),float(box.y1)]],
#             shape_type= "rectangle",
#             flags={},
#             group_id= None
#         )
#         for box in bbs_aug_move
#     ]
#     with open(out_json,'w') as outfile:
#         json.dump(data, outfile)
#
# BOX_COLOR = (255, 0, 0) # Red
# TEXT_COLOR = (255, 255, 255) # White

# def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
#     """Visualizes a single bounding box on the image"""
#     x_min, y_min, x_max, y_max, label = bbox
#     w = x_max - x_min
#     h = y_max - y_min
#     x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
#
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
#
#     (text_width, text_height) = (5,6)
#     cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
#     cv2.putText(
#         img,
#         text=label,
#         org=(x_min, y_min - int(0.3 * text_height)),
#         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#         fontScale=0.35,
#         color=TEXT_COLOR,
#         lineType=cv2.LINE_AA,
#     )
#     return img
#
# def visualize(image, bboxes):
#     img = image.copy()
#     for bbox in bboxes:
#         img = visualize_bbox(img, bbox)
#     plt.figure(figsize=(12, 12))
#     plt.axis('off')
#     plt.imshow(img)
#     plt.pause(50)


def checkBox(bboxes, w, h):
    new_boxes= []
    for i, box in enumerate(bboxes):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        new_box = []
        # if (x1 in range(0,w)) and (x2 in range(0,w)) and (y1 in range(0,h)) and (y2 in range(0,h)):
        if x1 == x2 or y1 == y2:
            print(box)
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
    # img_url = '../Wheat_rust_severity/source/result/yolov3_train_val/DJI_00026.JPG'
    # index = 1
    # out_dir = "../Wheat_rust_severity/source/result/yolov3_train_val/"
    # transform_crop_rot = A.Compose([
        # A.CropAndPad(percent=(-0.5,0.1)),
        # A.RandomSizedBBoxSafeCrop(height=500,width=1000)
        # A.crops.transforms.RandomCropFromBorders()
        #
        # A.CenterCrop(height=3000,width=3000)
        # A.Affine(rotate=14,rotate_method='ellipse', p=1),
        # A.Crop(x_min=400,y_min=100,x_max=5000,y_max=3100)
        # A.Transpose(p=0.5)
    # ], bbox_params=A.BboxParams(format='pascal_voc',min_visibility=0.5))
    # transform_name = "affineandCrop"
    # jsonBoxrotate(img_url,index=index,outdir=out_dir,transform=transform_crop_rot, transform_name=transform_name)







