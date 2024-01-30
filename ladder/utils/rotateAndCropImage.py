import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os

from skimage import io as skio
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops, label
from skimage.transform import rotate


from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000

def patches():
    img_url = './data/WW_20200625_11_rotated.png'
    map_url = '../raw/2020/map_winter_24_variety.csv'

    with open(map_url, 'r', encoding='utf-8-sig') as f:
        maps = np.genfromtxt(f, dtype=int, delimiter=',')

    rows, cols = maps.shape


    img = skio.imread(img_url)


    w_b, h_b = 1350, 320
    gap_row, gap_col = 140, 340
    top = 50
    for i in range(rows):
        left = 1
        for j in range(cols):
            print((left,top))
            cv2.circle(img,(left,top),30,(0,0,255),-1)
            # crop_img = img[top:top+h_b,left:left+w_b]
            # crop_img_url = './data/patch/' + str(maps[i,j]) + '.png'
            # plt.imsave(crop_img_url, crop_img)



            if j < range(cols)[-1]:

                left = left + w_b + gap_col
            else:
                pass
        if i < range(rows)[-1]:
            top = top + h_b + gap_row
        else:
            pass

    cv2.namedWindow("source image",cv2.WINDOW_NORMAL)
    cv2.resizeWindow('source image', 2000, 1000)
    cv2.imshow('source image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# rotate image by cv2 perspective transform
def rotateImg(img, pts):
    """
    Perspectively project assigned area (pts) to a rectangle image
    -----
    param.
    -----
    img: 2-d numpy array
    pts: a vector of xy coordinate, length is 4. Must be in the order as:
         (NW, NE, SE, SW)
    """

    # define input coordinates
    # pts = np.float32(pts)

    # assign sorted pts
    # pt_NW, pt_NE, pt_SE, pt_SW = sortPts(pts)
    pt_NW = pts[0]
    pt_NE = pts[1]
    pt_SE = pts[2]
    pt_SW = pts[3]


    # estimate output dimension
    img_W = (sum((pt_NE-pt_NW)**2)**(1/2)+sum((pt_SE-pt_SW)**2)**(1/2))/2
    img_H = (sum((pt_SE-pt_NE)**2)**(1/2)+sum((pt_SW-pt_NW)**2)**(1/2))/2

    shape = (int(img_W), int(img_H))
    print(shape)

    # generate target point
    pts2 = np.float32(
        # NW,    NE,            SE,                   SW
        [[0, 0], [shape[0], 0], [shape[0], shape[1]], [0, shape[1]]])

    print(pts2)
    pts = np.float32(np.row_stack(pts))
    # transformation
    H = cv2.getPerspectiveTransform(pts, pts2)
    dst = cv2.warpPerspective(img, H, (shape[0], shape[1]))
    dst = np.array(dst).astype(np.uint8)

    # return cropped image and H matrix
    return dst

class coordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, " ", y)
            self.points.append((x,y))


if __name__ == '__main__':
    # fold_url
    In_fd = './data_raw/'

    Out_fd ='./data_tile/'

    Out_fd_big = './data_rotate/'

    sub_fds = os.listdir(In_fd)
    for sub_fd in sub_fds:
        if sub_fd != '.DS_Store':
            sco_fd = sub_fd
            sub_fd = In_fd + sco_fd
            sub_imgs = os.listdir(sub_fd)
            out_sub_fd = Out_fd + sco_fd + '/'
            out_sub_fd_big = Out_fd_big + sco_fd + '/'

            for img in sub_imgs:
                if img != '.DS_Store':
                    source_img_url = sub_fd + '/' + img
                    source_img = skio.imread(source_img_url)

                    # get the coordinate for the first image in the folder
                    cv2.imshow('source image', source_img)
                    coors = coordinateStore()
                    cv2.setMouseCallback('source image', coors.select_point)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # apply the rotate
                    pts = []
                    for pt in coors.points:
                        x, y = pt
                        pt = np.array([x,y])
                        pts.append(pt)
                    print(pts)
                    rotated_img = rotateImg(source_img, pts)
                    rotated_img_url = out_sub_fd_big + img.replace('.png', '_') + "rotate" + ".png"
                    plt.imsave(rotated_img_url, rotated_img)

