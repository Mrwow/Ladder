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
- use the SAHI algrithom to improve small object detection accuray, here is the link for SAHI (https://github.com/obss/sahi)



## Installation

### Step 1: Install anconda
Install anconda from this link, and create a python environment
(https://www.anaconda.com/download/success)
```
conda create --name py3819 python=3.8.19
conda activate py3819 
```
### Step
