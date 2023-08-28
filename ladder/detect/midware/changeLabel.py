import os
import json

def changelableInYoloFormat(dir):
    for path, subdirs, fls in os.walk(dir):
        for fl in fls:
            if not fl.startswith("."):
                f_url = os.path.join(path,fl)
                print(f_url)
                new_file = []
                with open(f_url) as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.split(' ')
                    print(line)
                    if int(line[0]) < 1:
                        line[0] = "0"
                        print("===============")
                    elif int(line[0]) <= 5:
                        line[0] = "1"
                        print("(((((((((((((((((((")
                    elif int(line[0]) <=9:
                        line[0] = "2"
                        print("++++++++++++++++")
                    else:
                        line[0] = "3"
                        print(' '.join(line))

                    new_file.append(line)

                with open(f_url,'w+') as f:
                    for line in new_file:
                        f.write(' '.join(line))
    return

def changeJsonLabletoJson(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    for shape in data["shapes"]:
        score = int(shape["label"])
        print(score)
        if score == 0:
            print("No_rust")
            shape["label"] = str(0)
        elif score > 0 and score <= 20:
            print("Low")
            shape["label"] = str(1.0)
        elif score > 20 and score <= 60:
            print("Moderate")
            shape["label"] = str(2.0)
        elif score >60  and score <= 100:
            print("High")
            shape["label"] = str(3.0)
        else:
            print(score)

    a = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/ob/test/DJI_00048_2.json"
    with open(filename, 'w') as f:
        json.dump(data, f)




if __name__ == '__main__':
    # dir = "./labels"
    # changelableInYoloFormat(dir)

    fl ="/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/ob/test/DJI_00048.json"
    changeJsonLabletoJson(fl)