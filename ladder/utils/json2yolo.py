import json
from shutil import copy2
import os
import yaml

def jsonToYolo(input_path):
    label_list = []
    for f in os.listdir(input_path):
        if f.endswith("json"):
            filename = os.path.join(input_path,f)
            print(filename)
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                imagePath = data["imagePath"]
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
            # copy images into train data folder
            image_output_path = os.path.join(input_path,"train/images")
            imagePath = os.path.join(input_path,imagePath)
            if not os.path.exists(image_output_path):
                os.makedirs(image_output_path)
            copy2(imagePath,image_output_path)

            # label files folder
            base_name = os.path.basename(filename)
            label_name = base_name.replace('json','txt')
            labels_output_path = os.path.join(input_path,"train/labels/")
            if not os.path.exists(labels_output_path):
                os.makedirs(labels_output_path)
            label_name = os.path.join(labels_output_path,label_name)

            with open(label_name,'w') as f:
                for s in shapes:
                    label_list.append(s["label"]) if s["label"] not in label_list else label_list
                    print(label_list)
                    label_index = label_list.index(s["label"])
                    x1,y1, x2, y2=s["points"][0][0],s["points"][0][1],s["points"][1][0],s["points"][1][1]
                    w = (x2-x1)/img_w
                    h = (y2-y1)/img_h
                    x = (x1 + x2)/(2*img_w)
                    y = (y1 + y2)/(2*img_h)
                    f.write(f'{label_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
    # label summary file
    label_summary = os.path.join(input_path,"labels_summary.txt")
    with open(label_summary,"w") as f:
        for item in label_list:
            f.write(f'{item}\n')

    # generate train_data.yaml
    names = {}
    for i,item in enumerate(label_list):
        names[i] = item

    data_yaml = os.path.join(input_path,"train_data.yaml")
    train_data_config = {
        'path': input_path,
        'train': "train",
        'val': 'train',
        "names": names
    }
    with open(data_yaml, 'w', encoding="utf-8") as f:
        yaml.dump(train_data_config,f)


    data_dict = dict(
        data = os.path.join(input_path,'train'),
        names = label_list
    )
    return data_dict