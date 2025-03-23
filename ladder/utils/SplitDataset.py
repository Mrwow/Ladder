from PIL import Image
import os
import random
import shutil
from shutil import copy2
import pandas as pd


def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0, test_scale=0.2):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹 './experiment05_stage2/src'
    :param target_data_folder: 目标文件夹 './experiment05_stage2/src_split'
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    if os.path.exists(target_data_folder):
        shutil.rmtree(target_data_folder)
    os.makedirs(target_data_folder)
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            if not class_name.startswith('.'):
                class_split_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_split_path):
                    pass
                else:
                    os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        if not class_name.startswith('.'):
            current_class_data_path = os.path.join(src_data_folder, class_name)
            current_all_data = os.listdir(current_class_data_path)
            current_data_length = len(current_all_data)
            current_data_index_list = list(range(current_data_length))
            random.shuffle(current_data_index_list)

            train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
            val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
            test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
            train_stop_flag = current_data_length * train_scale
            val_stop_flag = current_data_length * (train_scale + val_scale)
            current_idx = 0
            train_num = 0
            val_num = 0
            test_num = 0
            for i in current_data_index_list:
                src_img_path = os.path.join(current_class_data_path, current_all_data[i])
                if current_idx <= train_stop_flag:
                    copy2(src_img_path, train_folder)
                    # print("{}复制到了{}".format(src_img_path, train_folder))
                    train_num = train_num + 1
                elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                    copy2(src_img_path, val_folder)
                    # print("{}复制到了{}".format(src_img_path, val_folder))
                    val_num = val_num + 1
                else:
                    copy2(src_img_path, test_folder)
                    # print("{}复制到了{}".format(src_img_path, test_folder))
                    test_num = test_num + 1

                current_idx = current_idx + 1

            print("*********************************{}*************************************".format(class_name))
            print(
                "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale,
                                                      current_data_length))
            print("训练集{}：{}张".format(train_folder, train_num))
            print("验证集{}：{}张".format(val_folder, val_num))
            print("测试集{}：{}张".format(test_folder, test_num))


def cropImg(input, h=224, w=224, output="./"):
    """crop raw image with defined size

    :param input: image path
    :param h:
    :param w:
    :param output:
    :return:
    """
    im = Image.open(input)
    im_name = input.split("/")[-1]
    im_name = im_name.replace('.JPG','')
    im_name = im_name.replace(".jpg",'')
#     print(im_name)
    W, H = im.size
    for i in range(0,H,h):
        raw_n = int(i/224 + 1)
        for j in range(0,W,w):
            col_n = int(j/224 + 1)
            if j+w <= W and i+h <= H:
                box = (j, i, j+w, i+h)
                tile = im.crop(box)
                tile.save(output + "/%s_%s_%s.png" % (im_name, raw_n, col_n))

def cropPatch(fd_src, out_src):
    imgs = os.listdir(fd_src)
    for img in imgs:
        if img != ".DS_Store":
            print(img)
            img_src = fd_src + "/" + img
            print(img_src)
            print(out_src)
            cropImg(input = img_src, output=out_src)


def cropSrcSplitWithLabel(target,label, output='./tiles'):
    # built output folder structure
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)

    splits =["test", "train", "val"]
    cls = os.listdir(target+'/'+ "test")

    for split in splits:
        for cl in cls:
            if not os.path.exists(output + '/' + split + '/' + cl):
                os.makedirs(output + '/' + split + '/' + cl)

    # crop image and put in related output folder based on the label
    for split in splits:
        for cl in cls:
            img_fd = target+'/'+ split + "/" + cl
            imgs = os.listdir(img_fd)
            for img in imgs:
                if img != ".DS_Store":
                    img_name = img.replace('.JPG', '')
                    img_name = img_name.replace(".jpg", '')
                    img_src = img_fd + "/" + img
                    print(img_src)
                    im = Image.open(img_src)
                    print(img_name)
                    # label
                    csv = label + '/' + img_name + '_labeloutput_confidthres=0.5.csv'
                    if os.path.exists(csv):
                        print(csv)
                        df = pd.read_csv(csv)
                        labels = df['label']
                        row_num = max(df['row']) + 1
                        col_num = max(df['col']) + 1

                        # image size
                        rgbwidth, rgbheight = im.size
                        row_stepsize = int(rgbheight / row_num)
                        col_stepsize = int(rgbwidth / col_num)

                        # crop image
                        row_ind = 0
                        for row in range(0, rgbheight, row_stepsize):
                            if row + row_stepsize <= rgbheight:
                                col_ind = 0
                                for col in range(0, rgbwidth, col_stepsize):
                                    if col + col_stepsize <= rgbwidth:
                                        b_w = col_stepsize
                                        b_h = row_stepsize
                                        box = (col, row, col + b_w, row + b_h)
                                        tile = im.crop(box)
                                        # tile output based on label
                                        label_ind = row_ind * col_num + col_ind
                                        if labels[label_ind] == 1:
                                            tile_output = output + '/' + split + '/PST'
                                        if labels[label_ind] == 0:
                                            tile_output = output + '/' + split + '/PST_free'

                                        tile.save(tile_output + "/%s_%s_%s.png" % (img_name, row_ind, col_ind))
                                    col_ind = col_ind + 1
                            row_ind = row_ind + 1
                    else:
                        if cl == 'PST_free':
                            tile_output = output + '/' + split + '/PST_free'
                            cropImg(input = img_src, output=tile_output)



def split_raw_images(fd, output):
    imgs = os.listdir(fd)
    for img in imgs:
        if img != ".DS_Store":
            if img.endswith(".JPG") or img.endswith(".jpg"):
                # print(img)
                img_src = fd + "/" + img
                print(img_src)
                im = Image.open(img_src)
                img_name = img.replace('.JPG', '')
                img_name = img_name.replace(".jpg", '')
                csv = fd + '/' + img_name + '_labeloutput_confidthres=0.5.csv'
                if os.path.exists(csv):
                    print(csv)
                    df = pd.read_csv(csv)
                    labels = df['label']
                    row_num = max(df['row']) + 1
                    col_num = max(df['col']) + 1

                    # image size
                    rgbwidth, rgbheight = im.size
                    row_stepsize = int(rgbheight / row_num)
                    col_stepsize = int(rgbwidth / col_num)

                    # crop image
                    row_ind = 0
                    for row in range(0, rgbheight, row_stepsize):
                        if row + row_stepsize <= rgbheight:
                            col_ind = 0
                            for col in range(0, rgbwidth, col_stepsize):
                                if col + col_stepsize <= rgbwidth:
                                    b_w = col_stepsize
                                    b_h = row_stepsize
                                    box = (col, row, col + b_w, row + b_h)
                                    tile = im.crop(box)
                                    # tile output based on label
                                    label_ind = row_ind * col_num + col_ind
                                    if labels[label_ind] == 1:
                                        tile_output = output + '/PST'
                                    if labels[label_ind] == 0:
                                        tile_output = output + '/PST_free'

                                    tile.save(tile_output + "/%s_%s_%s.png" % (img_name, row_ind, col_ind))
                                col_ind = col_ind + 1
                        row_ind = row_ind + 1

if __name__ == '__main__':
    # base = "../experiment09_stage4Pre/"
    # src = base + 'src'
    # target = base + 'src_split'
    # label = base + 'label'
    # output = base + 'tile'
    # data_set_split(src, target)
    # cropSrcSplitWithLabel(target, label, output)


    # base = "/Users/ZhouTang/Downloads/zzlab/1_Project/wheat_rust_HR/source/experiment11_addOtherDatasetInTraining/cornLeafDisease/corn/"
    # src = base + 'src'
    # target = base + 'src_split'
    # label = base + 'label'
    # output = base + 'tile'
    # data_set_split(src, target)
    # cropSrcSplitWithLabel(target, label, output)

    # data_set_split("/Users/ZhouTang/Downloads/zzlab/1_Project/wheat_rust_HR/source/experiment11_addOtherDatasetInTraining/del_mix_field/tile/allMixFieldData/FalsePositive",
    #                "/Users/ZhouTang/Downloads/zzlab/1_Project/wheat_rust_HR/source/experiment11_addOtherDatasetInTraining/del_mix_field/tile/allMixFieldData/FalsePositive_split")

    # fd_src = "/Users/ZhouTang/Desktop/SRDP/v4/stage1_trainingset_6/PST"
    # out_src = "/Users/ZhouTang/Desktop/SRDP/v4/stage1_trainingset_6/tile/PST"
    # cropPatch(fd_src, out_src)

    fd = "/Users/ZhouTang/Desktop/SRDP/v4/stage2_new/PST"
    output = "/Users/ZhouTang/Desktop/SRDP/v4/stage2_new/tile"
    split_raw_images(fd,output)

