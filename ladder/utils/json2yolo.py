import json
from shutil import copy2
import os
import random
import shutil
import glob

import cv2
import yaml
from pathlib import Path

from ladder.widgets.label_file import LabelFile
from .imgAug import grid2tileBatch
from .checkJson import checkBox_batch

IMG_EXTS = {'.heif', '.tif', '.heic', '.jpg', '.svg', '.cur', '.bmp', '.pgm', '.svgz', '.ppm', '.wbmp', '.ico', '.gif', '.icns', '.pbm', '.xbm', '.png', '.tga', '.jp2', '.webp', '.tiff', '.xpm', '.jpeg'}

def jsonToYolo(input_path):
    image_labeled = len(glob.glob(os.path.join(input_path, "*.json")))
    print(f'total {image_labeled} images have json file')
    checkBox_batch(input_path)
    if image_labeled <=10:
        jsonToYoloSameTrainTest(input_path=input_path)
    else:
        jsonToYoloTrainTestSplit(input_path=input_path)


def jsonToYoloSameTrainTest(input_path):
    label_list = []

    input_path = Path(input_path)
    base_fd = input_path
    yolo_root = base_fd / "yolo"

        # --- prepare output dirs ---
    for split in ("train","val","test"):
        (yolo_root / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_root / split / "labels").mkdir(parents=True, exist_ok=True)



    image_output_path = os.path.join(yolo_root,"train/images")
    # if not os.path.exists(image_output_path):
    #     os.makedirs(image_output_path)
    labels_output_path = os.path.join(yolo_root,"train/labels/")
    # if not os.path.exists(labels_output_path):
    #     os.makedirs(labels_output_path)

    for f in os.listdir(input_path):
        if f.endswith("jpg") or f.endswith("JPG") or f.endswith("png") or f.endswith("jpeg") or f.endswith("gif") \
            or f.endswith("heic") or f.endswith("jp2") or f.endswith("bmp") or f.endswith("heif") or f.endswith("tif") or f.endswith("tiff"):
            # copy image
            imagePath = os.path.join(input_path,f)

            imagePath = os.path.join(input_path,imagePath)
            copy2(imagePath,image_output_path)
            # print(f"img is {f}")

            # create txt file for yolo
            img = os.path.basename(f)
            img_json = img.split(".")[0] + ".json"
            img_json_url = os.path.join(input_path,img_json)

            if os.path.exists(img_json_url):
                print(f"{img} labeled json {img_json} is converted to a yolo txt")
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
                        # print(label_list)
                        label_index = label_list.index(s["label"])
                        x1,y1, x2, y2=s["points"][0][0],s["points"][0][1],s["points"][1][0],s["points"][1][1]
                        w = (x2-x1)/img_w
                        h = (y2-y1)/img_h
                        x = (x1 + x2)/(2*img_w)
                        y = (y1 + y2)/(2*img_h)
                        f.write(f'{label_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
            else:
                print(f"create a empty txt file for {img}")
                # label files folder
                label_name = img_json.replace('json','txt')
                labels_output_path = os.path.join(input_path,"train/labels/")
                if not os.path.exists(labels_output_path):
                    os.makedirs(labels_output_path)
                label_name = os.path.join(labels_output_path,label_name)
                with open(label_name,'w') as f:
                    pass

    # label summary file
    # label_summary = os.path.join(input_path,"train","labels_summary.txt")
    # # with open(label_summary,"w") as f:
    #     for item in label_list:
    #         f.write(f'{item}\n')


    # --- write YOLO data.yaml (point to images subfolders) ---
    data_yaml = yolo_root / "yolo_train_config.yaml"
    names = {}
    for i,item in enumerate(label_list):
        names[i] = item
    train_data_config = {
        'path': str(yolo_root),
        'train': "train/images",
        'val': "train/images",
        "names": names
    }
    with open(data_yaml, 'w', encoding="utf-8") as f:
        yaml.dump(train_data_config,f)

    data_dict = dict(
        data = data_yaml,
        names = label_list
    )
    return data_dict



def jsonToYoloTrainTestSplit(
    input_path,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    random_seed=42
):
    """
    Build a YOLO dataset directly from the original images and LabelMe JSONs
    WITHOUT tiling. For each split, copies images to <yolo>/<split>/images
    and writes YOLO .txt labels to <yolo>/<split>/labels.

    Assumes one JSON per image: <stem>.json next to image.
    Handles rectangles (2-point) and polygons (>=3 points) by using a tight bbox.
    """

    input_path = Path(input_path)
    base_fd = input_path
    yolo_root = base_fd / "yolo"

    # --- validate ratios ---
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0 (got {total})")

    # --- collect (image,json) pairs & label names ---
    # IMG_EXTS = {".jpg",".jpeg",".png",".gif",".bmp",".tif",".tiff",".heic",".heif",".jp2",".JPG",".PNG",".JPEG",".TIF",".TIFF"}
    pairs = []            # [(img_path, json_path)]
    label_set = set()

    for name in os.listdir(input_path):
        if name.startswith("."):
            continue
        p = input_path / name
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            j = input_path / (p.stem + ".json")
            if not j.exists():
                # skip images without JSON to keep pairs aligned
                continue
            # harvest labels
            try:
                with j.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for s in data.get("shapes", []):
                    lbl = s.get("label")
                    if lbl is not None:
                        label_set.add(lbl)
            except Exception:
                # skip malformed JSONs
                continue
            pairs.append((p, j))

    if not pairs:
        raise RuntimeError("No valid (image, json) pairs found in input folder.")

    # --- stable class ordering & maps ---
    labels_sorted = sorted(label_set)
    label_to_id = {lbl: i for i, lbl in enumerate(labels_sorted)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    # --- prepare output dirs ---
    for split in ("train","val","test"):
        (yolo_root / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_root / split / "labels").mkdir(parents=True, exist_ok=True)

    # --- write YOLO data.yaml (point to images subfolders) ---
    data_yaml = yolo_root / "yolo_train_config.yaml"
    train_data_config = {
        "path": str(yolo_root),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "names": id_to_label,
    }
    with data_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(train_data_config, f, sort_keys=False, allow_unicode=True)

    # --- deterministic shuffle & split ---
    rng = random.Random(random_seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    blocks = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train+n_val],
        "test":  pairs[n_train+n_val:],
    }

    # --- helper: write one YOLO txt from a LabelMe shapes list ---
    def write_yolo_from_shapes(shapes, img_w, img_h, out_txt_path):
        with open(out_txt_path, "w", encoding="utf-8") as fh:
            for s in shapes:
                lbl = s.get("label")
                pts = s.get("points", [])
                if lbl not in label_to_id or len(pts) < 2:
                    continue

                # If rectangle: first two points; if polygon: use all points
                xs = [pt[0] for pt in pts]
                ys = [pt[1] for pt in pts]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # normalize to YOLO
                w = (x_max - x_min) / float(img_w)
                h = (y_max - y_min) / float(img_h)
                cx = (x_min + x_max) / (2.0 * img_w)
                cy = (y_min + y_max) / (2.0 * img_h)

                # skip degenerate boxes
                if w <= 0 or h <= 0:
                    continue

                cls_id = label_to_id[lbl]
                fh.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # --- copy images & generate labels (no tiling) ---
    for split, items in blocks.items():
        images_dir = yolo_root / split / "images"
        labels_dir = yolo_root / split / "labels"

        for img_p, json_p in items:
            # copy image
            copy2(str(img_p), str(images_dir))

            # convert JSON → YOLO .txt
            try:
                with open(json_p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                img_w = data["imageWidth"]
                img_h = data["imageHeight"]
                shapes = data.get("shapes", [])
            except Exception:
                # skip if broken JSON
                continue

            txt_name = img_p.stem + ".txt"
            out_txt = labels_dir / txt_name
            write_yolo_from_shapes(shapes, img_w, img_h, out_txt)

    return {"data_url": str(data_yaml), "names": labels_sorted}


    # # create folder for "train", "val", "test"
    # fold_list = ["train", "val", "test"]
    # base_fd = os.path.dirname(input_path)
    # for fd in fold_list:
    #     # fd_img = os.path.join(base_fd,"yolo",fd)
    #     fd_img = os.path.join(input_path,"train_test", fd)
    #     if not os.path.exists(fd_img):
    #         os.makedirs(fd_img)
    #     # fd_json = os.path.join(base_fd,"yolo",fd)
    #     fd_json = os.path.join(input_path,"train_test",fd)
    #     if not os.path.exists(fd_json):
    #         os.makedirs(fd_json)

    # # get all image name, json file name, and label list
    # list_image = []
    # list_label = []
    # list_json = []
    # for f in os.listdir(input_path):
    #     if f.endswith("jpg") or f.endswith("JPG") or f.endswith("png") or f.endswith("jpeg") or f.endswith("gif") \
    #         or f.endswith("heic") or f.endswith("jp2") or f.endswith("bmp") or f.endswith("heif") or f.endswith("tif") or f.endswith("tiff"):
    #     # if not f.startswith(".") and (f.endswith("jpg") or f.endswith("JPG") or f.endswith("png")) or f.endswith("jpeg"): 
    #         list_image.append(f)
    #         img = os.path.basename(f)
    #         img_json = img.split(".")[0] + ".json"
    #         img_json_url = os.path.join(input_path,img_json)
    #         list_json.append(img_json_url)
    #         if os.path.exists(img_json_url):
    #             try:
    #                 with open(img_json_url, "r") as f:
    #                     data = json.load(f)
    #                 shapes = [
    #                     dict(
    #                         label=s["label"],
    #                     )
    #                     for s in data["shapes"]
    #                 ]
    #                 for s in shapes:
    #                     if s["label"] not in list_label:
    #                         list_label.append(s["label"])
    #             except Exception as e:
    #                 pass

    # # generate yolo_train_config.yaml
    # sorted_list_label = list(enumerate(list_label))
    # sorted_list_label = sorted(sorted_list_label, key=lambda x: x[1])
    # dict_id_label = {}
    # dict_label_id = {}
    # for i in range(len(sorted_list_label)):
    #     dict_id_label[i] = sorted_list_label[i][1]
    #     dict_label_id[sorted_list_label[i][1]] = i

    # data_yaml = os.path.join(base_fd, "yolo","yolo_train_config.yaml")
    # train_data_config = {
    #     'path': os.path.join(base_fd,"yolo"),
    #     'train': "train",
    #     'val': 'val',
    #     'test':'test',
    #     "names": dict_id_label
    # }
    # with open(data_yaml, 'w', encoding="utf-8") as f:
    #     yaml.dump(train_data_config,f)

    # # copy image and json file
    # indx_img_list = list(range(len(list_image)))
    # len_imgs = len(indx_img_list)
    # random.seed(random_seed)
    # random.shuffle(indx_img_list)
    # train_stop_flag = len_imgs * train_ratio
    # val_stop_flag = len_imgs * (train_ratio + val_ratio)
    # current_idx = 0
    # train_num = 0
    # val_num = 0
    # test_num = 0
    # for i in indx_img_list:
    #     url_img = os.path.join(input_path,list_image[i])
    #     url_json = os.path.join(input_path, list_json[i])
    #     json_file_name = os.path.basename(url_json)
    #     if current_idx <= train_stop_flag:
    #         targt_img = os.path.join(base_fd,"yolo","train")
    #         targt_json = os.path.join(base_fd,"yolo","train",json_file_name)
    #         train_num = train_num + 1
    #     elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
    #         targt_img = os.path.join(base_fd,"yolo","val")
    #         targt_json = os.path.join(base_fd,"yolo","val",json_file_name)
    #         val_num = val_num + 1
    #     else:
    #         targt_img = os.path.join(base_fd,"yolo","test")
    #         targt_json = os.path.join(base_fd,"yolo","test",json_file_name)
    #         test_num = test_num + 1
        
    #     copy2(url_img, targt_img)
    #     copy2(url_json,targt_json)
    #     current_idx = current_idx + 1

    
    # for fd in fold_list:
    #     fd_img_folder = os.path.join(base_fd,"yolo",fd)
    #     sub_img_fd = os.path.join(fd_img_folder,"images")
    #     sub_yolo_fd = os.path.join(fd_img_folder,"lables")
    #     if not os.path.exists(sub_img_fd):
    #         os.makedirs(sub_img_fd)
    #     if not os.path.exists(sub_yolo_fd):
    #         os.makedirs(sub_yolo_fd)

    #     # get image patch
    #     grid2tileBatch(fd=fd_img_folder, grid_size=1200, min_visi=0.60)

    #     fd_img_grid = os.path.join(fd_img_folder,"grids")
    #     # create yolo txt for each image
    #     for f in os.listdir(fd_img_grid):
    #         if f.endswith('json') and not f.startswith('.'):
    #             tile_name = f.replace(".json",".JPG")
    #             tile_url = os.path.join(fd_img_grid,tile_name)
    #             copy2(tile_url, sub_img_fd)

    #             tile_json = os.path.join(fd_img_grid,f)
    #             txt_name = f.replace(".json",".txt")
    #             tile_txt = os.path.join(sub_yolo_fd, txt_name)

    #             try:
    #                 with open(tile_json, "r") as f:
    #                     data = json.load(f)
    #                 # imagePath = data["imagePath"]
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

    #             # yolo txt output
    #             print(f"conver {tile_json} to {tile_txt}")
    #             with open(tile_txt,'w') as f:
    #                 for s in shapes:
    #                     label_index = dict_label_id[s["label"]]
    #                     x1,y1, x2, y2=s["points"][0][0],s["points"][0][1],s["points"][1][0],s["points"][1][1]
    #                     w = (x2-x1)/img_w
    #                     h = (y2-y1)/img_h
    #                     x = (x1 + x2)/(2*img_w)
    #                     y = (y1 + y2)/(2*img_h)
    #                     f.write(f'{label_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

    # data_dict = dict(
    #     data_url = data_yaml,
    #     names = list_label
    # )
    
    # return data_dict



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

