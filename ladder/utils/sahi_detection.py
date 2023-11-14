from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict
import cv2
import json
import os


def coco2json(coco,img_url):

    img= cv2.imread(img_url)
    im_h,im_w,c_ = img.shape
    shapes =[]

    img_name =os.path.basename(img_url)
    print(img_name)
    print(f"img_w {im_w}, img_h {im_h}")
    dir = os.path.dirname(img_url)

    for detection in coco:
        bbox = detection['bbox']
        score = detection['score']
        category_id = detection['category_id']
        category_name = detection['category_name']
        area = detection['area']
        x1, y1 = bbox[0], bbox[1]
        w, h = bbox[2], bbox[3]
        x2 = x1 + w
        y2 = y1 + h
        shape = dict(
            points = [[x1,y1],[x2,y2]],
            label = category_name,
            score = score,
            shape_type = "rectangle",
            flags = {},
            group_id =None,
            other_data = {
                "category_id":category_id,
                "area": area
            },
        )
        shapes.append(shape)

    data_out = dict(
        version="5.0.2",
        flags={},
        shapes=shapes,
        imagePath=img_name,
        imageData=None,
        imageHeight=im_w,
        imageWidth=im_h,
    )
    json_out = img_name.split(".")[0] + ".json"
    json_out_url = os.path.join(dir, json_out)
    with open(json_out_url, "w") as f:
        json.dump(data_out, f, ensure_ascii=False, indent=2)

    return data_out

def sliceDetectBatch(weight,img_fd,conf,iou,img_size,img_h,img_w,overlap):
    # batch mode
    imgs = os.listdir(img_fd)
    for img in imgs:
        print(img)
        if not img.startswith(".") and img.split(".")[1] in ['png', 'jpg', 'JPG', 'jepg', 'JEPG']:
            img_url = os.path.join(img_fd,img)
            print(img_url)
            sliceDetect(weight=weight,img=img_url,conf=conf,iou=iou,
                        img_size=img_size,img_h=img_h,img_w=img_w,overlap=overlap)



def sliceDetect(weight,img,conf,iou,img_size,img_h,img_w,overlap):
    # SAHI sliced
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=weight,
        confidence_threshold=conf,
        device="cpu", # or 'cuda:0'
        image_size=img_size

    )
    # dir = os.path.dirname(img)
    result = get_sliced_prediction(
        img,
        detection_model,
        slice_height = img_h,
        slice_width = img_w,
        overlap_height_ratio = overlap,
        overlap_width_ratio = overlap,
        postprocess_match_threshold=iou
    )
    coco = result.to_coco_annotations()
    coco2json(coco,img_url=img)

# if __name__ == '__main__':
#     weight = "/Users/ZhouTang/Downloads/zzlab/1_Project/ladder/source/data/wheat_seed/weights/best.pt"
#     img = "/Users/ZhouTang/Downloads/zzlab/1_Project/ladder/source/data/wheat_seed/test/BK_V19#5_S.jpg"
#     sliceDetect(weight=weight,img=img,conf=0.1,iou=0.2,img_size=1200,img_h=2400, img_w=2400, overlap=0.4)




