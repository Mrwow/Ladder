import json
import os
from PIL import Image


def checkBox(box, w, h):
    """
    box: [[x1,y1],[x2,y2]]
    """
    print(f"box is {box}")
    if len(box) == 2 : 
        x1 = box[0][0]
        y1 = box[0][1]
        x2 = box[1][0]
        y2 = box[1][1]

        # check is out of the canvas
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2, w)
        y2 = min(y2, h)
        box = [[x1,y1],[x2,y2]]

        new_box = []
        # check if in good order
        if x1 == x2 or y1 == y2:
            pass
        elif x1 < x2 and y1 < y2:
            new_box = box
        elif x1 > x2 and y1 > y2:
            new_box = [[x2,y2],[x1,y1]]
        elif x1 < x2 and y1 > y2:
            new_box = [[x1,y2],[x2,y1]]
        elif x1 > x2 and y1 < y2:
            new_box = [[x2,y1],[x1,y2]]
    elif len(box) == 4:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2, w)
        y2 = min(y2, h)
        box = [[x1,y1],[x2,y2]]
        new_box = []
        if x1 == x2 or y1 == y2:
            pass
        elif x1 < x2 and y1 < y2:
            new_box = box
        elif x1 > x2 and y1 > y2:
            new_box = [[x2,y2],[x1,y1]]
        elif x1 < x2 and y1 > y2:
            new_box = [[x1,y2],[x2,y1]]
        elif x1 > x2 and y1 < y2:
            new_box = [[x2,y1],[x1,y2]]
    else:
        pass
    print(f"new box is {new_box}")
    return new_box

def checkBoxImg(box, img):
    """
    box: [[x1,y1],[x2,y2]]
    """
    w, h = Image.open(img).size
    new_box = checkBox(box=box,w=w,h=h)
    return new_box


def checkBox_batch(fd):
    for f in os.listdir(fd):
        if f.endswith(".json") and not f.startswith("."):
            print(f)
            json_url = os.path.join(fd, f)
            with open(json_url, "r") as f_json:
                data = json.load(f_json)
            h = data["imageHeight"]
            w = data["imageWidth"]

            for shape in data["shapes"]:
                box = shape["points"]
                print(box)
                shape["points"] = checkBox(box=box,w=w,h=h)
            
            with open(json_url,'w') as outfile:
                json.dump(data, outfile)
    return


if __name__ == '__main__':
    # fd = "/Volumes/work_Joe/archive/2024/WSU/ladder/app/Alfalfa/data/solidStem/exp02/all"
    fd = "/Volumes/work_Joe/archive/ladder/source/data/debug/json_loc"
    checkBox_batch(fd)