# Ladder: A Software to Label Images, Detect Objects and Deploy Models Recurrently for Object Detection
This is a software that allow users to label image, train YOLO, and detect images in single GUI. Here are main functions in Ladder. 

## Introduction of main functions

### Label images with bboxes
- Draw rectangle and add lables
- Adjust size, label and postion of rectangles
- Save shapes and labels into a JSON file
- Reload the JSON file into the Ladder and resume labeling work
- Check the JSON file format
- Conver JSON file to YOLO format

### Training with labeled data
- Currently support YOLO model set from the ultralytics (https://www.ultralytics.com/). Thanks them for the great work!
- Support split images into train, validation, and testing set if you labeled a lot of image, like 10 images. Otherwise, the testing set will be same as training set.
- Default image size and traing epoch can be edit as needed.

### Prediction or detection
- Use the SAHI algrithom to improve small object detection accuray, here is the link for SAHI (https://github.com/obss/sahi). Thanks them for the great work!
- Adjust different condifence and IoU as needed.
- Detection in a single image or in a folder with multiple image

## Installation

### Step 1: Install anconda
Install anconda from this link, and create a python environment
(https://www.anaconda.com/download/success)
```
conda create --name py3819 python=3.8.19
conda activate py3819 
```
### Step 2: Install third party python packages
some packages can be install with `pip install`, some packages can be install with `conda install`. Please be patient during installation. This is python daily common.
- For number and plot
```
pip install opencv-python==4.9.0.80
conda install numpy pandas matplotlib Pillow seaborn scipy
```
- For GUI
```
pip install qtpy==2.4.1 PyQt5==5.15.10 tqdm==4.66.4 imgviz==1.7.5 grpcio==1.64.1

conda install -c conda-forge pyside2
```
- For deep learning
```
pip install ultralytics==8.2.39 albumentations==1.4.10 torch==1.13.1 torchvision==0.14.1 tensorboard==2.14.0 sahi==0.11.16 albucore==0.0.12
```

### Step 3: Install the Ladder

You need first download the code and unzip it. Then use `cd ladder` to change the foler. Finally, you can install the ladder in your computer.I recommond use `develop` mode during the installation where Ladder is not actually installed in your computer. Each time you download the new version of Ladder, just replace with new code in the older and can go to step 4 directly.

```
python setup.py develop
```
or if you don't care, please use this command below
```
python setup.py install
```
if you want to delete it, please use this command below
```
pip uninstall ladder
```

### Step 4: Launch the Ladder
Please 
```
python -m ladder
```


