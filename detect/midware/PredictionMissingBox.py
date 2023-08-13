import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import time
from .addBoxes import addBox

def resnet18(weight='./detect/test_rust_4cls.pth'):
    # define model
    model = models.resnet18(pretrained=False)
    num_fc_in = model.fc.in_features
    model.fc = nn.Linear(num_fc_in, 3)
    # load weight
    weights = torch.load(weight, map_location='cpu')
    model.load_state_dict(weights)
    return model

def prediction2ndNN(img0,shapes):
    labels = ['0','1','2']
    im = Image.open(img0)
    # model to cpu or gpu
    model = resnet18()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    y_pred = []
    new_shapes_label=[]
    model.eval()
    with torch.no_grad():
        for box in shapes:
            input_img = im.crop(box)
            input_tensor = transform(input_img)
            input_batch = input_tensor.unsqueeze(0)
            input_batch = input_batch.to(device)
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            probabilities = probabilities.tolist()
            label_index = probabilities.index(max(probabilities))
            y_pred.append(labels[label_index])
            new_shape = dict(
                label = labels[label_index],
                points= [[box[0],box[1]],[box[2],box[3]]],
                shape_type="rectangle",
                flags={},
                group_id=None,
                other_data = {}
            )
            new_shapes_label.append(new_shape)
    print(len(y_pred))
    print(y_pred)

    return new_shapes_label

def moveShape(filter_index,shapes):
    print(filter_index)
    print(len(shapes))
    for j in sorted(filter_index,reverse=True):
        del shapes[j]
    print(len(shapes))
    return shapes


def imputeMissingBoxes(img,shapes):
    new_shapes, filter_index = addBox(shapes,im0_w=5472, im0_h=3648)
    print("new shapes")
    print(len(new_shapes))
    new_shapes_label = prediction2ndNN(img0=img,shapes=new_shapes)
    print(new_shapes_label)
    shapes = moveShape(filter_index, shapes)
    shapes = shapes + new_shapes_label

    return shapes

if __name__ == '__main__':
    file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/addBox/DJI_00033.json"
    # file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/addBox/DJI_00040.json"
    # file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/addBox/DJI_00018.json"
    shapes = addBox(file)
    img0 = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/addBox/DJI_00033.JPG"
    prediction2ndNN(img0,shapes)