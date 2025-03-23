# Ladder: A Software to Label Images, Detect Objects and Deploy Models Recurrently for Object Detection
This is a software that allow users to label image, train YOLO, and detect images in single GUI. Here are main functions in Ladder. 

## Introduction of main functions

### Label images with bboxes
- Draw rectangle and add lables
- Adjust size, label and postion of rectangles
- save shapes and labels into a JSON file
- reload the JSON file into the Ladder

### Training with labeled data
- Currently support YOLO model set from the ultralytics (https://www.ultralytics.com/)

### Prediction or detection
- Use the SAHI algrithom to improve small object detection accuray, here is the link for SAHI (https://github.com/obss/sahi)


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
pip install qtpy==2.4.1  
pip install PyQt5==5.15.10
pip install tqdm==4.66.4 
pip install imgviz==1.7.5
pip install grpcio==1.64.1
conda install -c conda-forge pyside2
```
- For deep learning
```
pip install ultralytics==8.2.39
pip install albumentations==1.4.10
pip install torch==1.13.1
pip install torchvision==0.14.1
pip install tensorboard==2.14.0
pip install sahi==0.11.16
pip install albucore==0.0.12

pip install ultralytics==8.2.39 albumentations==1.4.10 torch==1.13.1 torchvision==0.14.1 tensorboard==2.14.0 sahi==0.11.16 albucore==0.0.12
```
