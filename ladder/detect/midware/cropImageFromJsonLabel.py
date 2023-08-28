import json
import os
from PIL import Image
import random
import cv2

def pad2squarePIL(img, pad_num=0):
    w, h = img.size
    print((w,h))
    if w == h:
        return img
    elif w > h:
        new_img = Image.new(img.mode, (w,w), (pad_num,pad_num,pad_num))
        new_img.paste(img,(0, (w-h)//2))
    else:
        new_img = Image.new(img.mode, (h,h), (pad_num,pad_num,pad_num))
        new_img.paste(img, ((h-w)//2, 0))

    return new_img

def cropImageWithJsonLabel(dir, subcl=False, pading=False):
    fls = os.listdir(dir)
    for f in fls:
        if f.find(".json") != -1:
            json_file = os.path.join(dir,f)
            with open(json_file) as f:
                jf = json.load(f)
                img = os.path.join(dir,jf["imagePath"])
                img = Image.open(img)
                for s in jf["shapes"]:
                    img_fd = os.path.join(dir,s["label"])
                    if not os.path.exists(img_fd):
                        os.makedirs(img_fd)
                    print(img_fd)
                    x0 = int(s["points"][0][0])
                    y0 = int(s["points"][0][1])
                    x1 = int(s["points"][1][0])
                    y1 = int(s["points"][1][1])
                    box = (x0,y0,x1,y1)
                    sub_img = img.crop(box)
                    if pading:
                        sub_img = pad2squarePIL(sub_img,pad_num=0)

                    sub_img_name = jf["imagePath"] + "_" + s["label"] + "_" + str(x0) + "_" + str(y0) + "_" + str(x1) + "_" + str(y1) + ".png"
                    sub_img_dir = os.path.join(img_fd,sub_img_name)
                    sub_img.save(sub_img_dir,quality=100)
                    if subcl:
                        score = int(s["label"])
                        if score <=20:
                        #     fd = os.path.join(dir, "No_rust")
                        # elif score > 0 and score <=20:
                            fd = os.path.join(dir, "Low")
                        elif score > 20 and score <= 60:
                            fd = os.path.join(dir,"Moderate")
                        elif score > 60 and score <= 100:
                            fd = os.path.join(dir, "Serious")
                        else:
                            pass
                        if not os.path.exists(fd):
                            os.makedirs(fd)
                        sub_img_dir_2 = os.path.join(fd,sub_img_name)
                        sub_img.save(sub_img_dir_2,quality=100)



def cropTile(img_url,out_dim=224,num=500):
    img = Image.open(img_url)
    w, h = img.size
    print(w,h)
    dim_min = min(w,h)
    dim_max = max(w,h)
    i = 0
    if w != h:
        while i < num:
            i = i + 1
            if dim_min >= out_dim:
                x0 = random.randint(0,w-224)
                y0 = random.randint(0,h-224)
                x1 = x0 + out_dim
                y1 = y0 + out_dim
                box = (x0,y0,x1,y1)
            else:
                z = random.randint(0,dim_max-dim_min)
                if w <= h:
                    box = (0, z, dim_min, z+dim_min)
                else:
                    box = (z, 0, z+dim_min, dim_min)
            sub_img = img.crop(box)
            sub_img_url = img_url + "_"+ str(box[0])+ "_"+ str(box[1])+ "_"+ str(box[2])+ "_"+ str(box[3]) + ".png"
            sub_img.save(sub_img_url,quality=100)


def cropTileBatch(dir, num=10):
    for path, subdirs, fls in os.walk(dir):
        for fl in fls:
            if not fl.startswith("."):
                img_url = os.path.join(path,fl)
                print(img_url)
                cropTile(img_url,num=num)
                os.remove(img_url)


if __name__ == '__main__':
    dir =  "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/signscore/score/f2"
    cropImageWithJsonLabel(dir,subcl=True,pading=True)
    #
    # dir =  "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/modeling/experiment01_spillmancv/spring_spillman"
    # cropImageWithJsonLabel(dir,subcl=True)

    # dir = "../modeling/experiment01_spillmancv/win_spillman_cv/f1/health"
    # cropTileBatch(dir, num=500)
    #
    # dir = "../modeling/experiment01_spillmancv/win_spillman_cv/f1/light"
    # cropTileBatch(dir, num=10)
    #
    # dir = "../modeling/experiment01_spillmancv/win_spillman_cv/f1/medium"
    # cropTileBatch(dir, num=20)
    #
    # dir = "../modeling/experiment01_spillmancv/win_spillman_cv/f1/serious"
    # cropTileBatch(dir, num=15)


    # dir = "../modeling/experiment01_spillmancv/win_spillman_cv/f2/health"
    # cropTileBatch(dir, num=500)
    #
    # dir = "../modeling/experiment01_spillmancv/win_spillman_cv/f2/light"
    # cropTileBatch(dir, num=10)
    #
    # dir = "../modeling/experiment01_spillmancv/win_spillman_cv/f2/medium"
    # cropTileBatch(dir, num=20)
    #
    # dir = "../modeling/experiment01_spillmancv/win_spillman_cv/f2/serious"
    # cropTileBatch(dir, num=10)


    # dir = "../modeling/experiment01_spillmancv/spring_spillman/health"
    # cropTileBatch(dir, num=200)
    #
    # dir = "../modeling/experiment01_spillmancv/spring_spillman/light"
    # cropTileBatch(dir, num=10)
    #
    # dir = "../modeling/experiment01_spillmancv/spring_spillman/medium"
    # cropTileBatch(dir, num=10)
    #
    # dir = "../modeling/experiment01_spillmancv/spring_spillman/serious"
    # cropTileBatch(dir, num=10)