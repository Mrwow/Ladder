from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict
import cv2
import json
import os
import torch
import argparse
from pathlib import Path
import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# Optional Qt fallback
# try:
#     from PySide6.QtGui import QImageReader, QImage
#     from PySide6.QtWidgets import QApplication
#     _HAS_QT = True
# except Exception:
#     try:
#         from PyQt6.QtGui import QImageReader, QImage
#         from PyQt6.QtWidgets import QApplication
#         _HAS_QT = True
#     except Exception:
#         _HAS_QT = False
try:
    from qtpy.QtGui import QImageReader, QImage
    from qtpy.QtWidgets import QApplication
    _HAS_QT = True
except Exception:
    _HAS_QT = False


from sahi.model import AutoDetectionModel


IMG_EXTS = {'.heif', '.tif', '.heic', '.jpg', '.svg', '.cur', '.bmp', '.pgm', '.svgz', '.ppm', '.wbmp', '.ico', '.gif', '.icns', '.pbm', '.xbm', '.png', '.tga', '.jp2', '.webp', '.tiff', '.xpm', '.jpeg'}


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

def sliceDetectBatch(weight,img_fd,conf,iou,img_size,img_h,img_w,overlap,gpu):
    # batch mode
    imgs = os.listdir(img_fd)
    for img in imgs:
        print(img)
        p = Path(img_fd) / img
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
        # if not img.startswith(".") and img.split(".")[1] in ['png', 'jpg', 'JPG', 'jepg', 'JEPG']:
            img_url = os.path.join(img_fd,img)
            sliceDetect(weight=weight,img=img_url,conf=conf,iou=iou,
                        img_size=img_size,img_h=img_h,img_w=img_w,overlap=overlap,gpu=gpu)


# def sliceDetect(weight,img,conf,iou,img_size,img_h,img_w,overlap,gpu):
#     # SAHI sliced
#     gpu = 'cuda:' + str(gpu)
#     device = gpu if torch.cuda.is_available() else "cpu"
#     detection_model = AutoDetectionModel.from_pretrained(
#         model_type='yolov8',
#         model_path=weight,
#         confidence_threshold=conf,
#         device=device, # or 'cuda:0'
#         image_size=img_size

#     )
#     # dir = os.path.dirname(img)
#     result = get_sliced_prediction(
#         img,
#         detection_model,
#         slice_height = img_h,
#         slice_width = img_w,
#         overlap_height_ratio = overlap,
#         overlap_width_ratio = overlap,
#         postprocess_match_threshold=iou
#     )
#     coco = result.to_coco_annotations()
#     coco2json(coco,img_url=img)

# Optional decoders


# ---- helper: robust image loader -> RGB uint8 numpy array or None ----
def _load_image_rgb_array(path_str: str):
    p = Path(path_str)
    # 1) PIL
    if _HAS_PIL:
        try:
            with Image.open(p) as im:
                # Handle images with alpha by dropping alpha to RGB
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
                if im.mode == "RGBA":
                    im = im.convert("RGB")
                arr = np.array(im, dtype=np.uint8)
                # Ensure HxWx3
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    return arr[:, :, :3]
        except Exception:
            pass

    # 2) OpenCV
    if _HAS_CV2:
        try:
            arr_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if arr_bgr is not None:
                # BGR -> RGB
                return cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

    # 3) Qt
    if _HAS_QT:
        try:
            app = QApplication.instance() or QApplication([])
            reader = QImageReader(str(p))
            qimg = reader.read()
            if not qimg.isNull():
                # Convert to 32-bit RGBA and then to numpy
                qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
                width = qimg.width()
                height = qimg.height()
                ptr = qimg.bits()
                ptr.setsize(qimg.byteCount())
                arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
                # Drop alpha -> RGB
                return arr[:, :, :3].copy()
        except Exception:
            pass

    # None of the decoders worked
    return None


# ---- your function, now robust to unsupported formats ----
def sliceDetect(weight, img, conf, iou, img_size, img_h, img_w, overlap, gpu):
    """
    SAHI sliced detection that gracefully skips unsupported/unreadable images
    (e.g., AVIF/WEBP when not supported), instead of crashing.
    - 'img' may be a path; this function will try to decode it to an RGB array.
    - If decoding fails, the function returns None and prints a one-line warning.
    """
    # Resolve device
    if torch.cuda.is_available():
        device = f"cuda:{gpu}" if isinstance(gpu, (int, str)) else "cuda:0"
    else:
        device = "cpu"

    # Build detection model once per process ideally; here kept inside for drop-in use
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=weight,
        confidence_threshold=conf,
        device=device,
        image_size=img_size,
    )

    # Try to load the image robustly; if path fails, skip gracefully
    if isinstance(img, (str, os.PathLike)):
        img_arr = _load_image_rgb_array(str(img))
        if img_arr is None:
            print(f"[sliceDetect] Skipping unreadable/unsupported image: {img}")
            return None
        image_for_sahi = img_arr  # SAHI accepts numpy arrays (HxWxC, RGB)
        img_url_for_meta = str(img)
    else:
        # Already an array? ensure uint8 RGB
        img_arr = np.asarray(img)
        if img_arr.ndim != 3 or img_arr.shape[2] < 3:
            print(f"[sliceDetect] Skipping non-RGB array input: {type(img)} {getattr(img_arr,'shape',None)}")
            return None
        if img_arr.dtype != np.uint8:
            img_arr = img_arr.astype(np.uint8, copy=False)
        image_for_sahi = img_arr[:, :, :3]
        img_url_for_meta = "<array>"

    try:
        result = get_sliced_prediction(
            image=image_for_sahi,
            detection_model=detection_model,
            slice_height=img_h,
            slice_width=img_w,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_match_threshold=iou,
        )
    except Exception as e:
        print(f"[sliceDetect] SAHI prediction failed for {img_url_for_meta}: {e}")
        return None

    # Export to COCO and your downstream JSON
    try:
        coco = result.to_coco_annotations()
    except Exception as e:
        print(f"[sliceDetect] Failed to convert result to COCO for {img_url_for_meta}: {e}")
        return None

    try:
        # your function that saves JSON; assumed to exist in your codebase
        coco2json(coco, img_url=img_url_for_meta)
    except Exception as e:
        print(f"[sliceDetect] coco2json failed for {img_url_for_meta}: {e}")
        return None

    return coco  # or return result if you prefer




# parser = argparse.ArgumentParser(description="Define the data folder, gpu and out name")
# parser.add_argument('--data', '-d', help='the data folder', required=True)
# parser.add_argument('--weight', '-w', help='the weight path', required=True)
# parser.add_argument('--conf', '-c',type=float, help='confidence', required=True)
# parser.add_argument('--iou', '-i',type=float, help='iour', required=True)
# parser.add_argument('--img', '-s',type=int, help='image size', required=True)
# parser.add_argument('--slice', '-l',type=int, help='slice size', required=True)
# parser.add_argument('--overlap', '-o', type=float, help='overlap', required=True)
# parser.add_argument('--gpu', '-g', type=int, help='overlap', required=True)
# args = parser.parse_args()
#
#
# if __name__ == '__main__':
#     sliceDetectBatch(weight=args.weight,
#                      img_fd=args.data,
#                      conf=args.conf,
#                      iou=args.iou,
#                      img_size=args.img,
#                      img_h=args.slice,img_w=args.slice,
#                      overlap=args.overlap, gpu=args.gpu)





# if __name__ == '__main__':
#     weight = "/Users/ZhouTang/Downloads/zzlab/1_Project/ladder/source/data/wheat_seed/weights/best.pt"
#     img = "/Users/ZhouTang/Downloads/zzlab/1_Project/ladder/source/data/wheat_seed/test/BK_V19#5_S.jpg"
#     sliceDetect(weight=weight,img=img,conf=0.1,iou=0.2,img_size=1200,img_h=2400, img_w=2400, overlap=0.4)




