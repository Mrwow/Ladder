import json

def sortbox(file, start_point="5_1"):

    with open(file, "r") as f:
        data = json.load(f)

    shapes = [
        dict(
            label=s["label"],
            points=s["points"]
        )
        for s in data["shapes"]
    ]
    points = [s["points"] for s in data["shapes"]]

    points_sum = list(map(lambda x:[sum(x[0]),x[0],x[1]],points))
    print(len(points_sum))
    final_sorted_list = []

    while True:
        try:
            # find the initial box
            new_sorted = []
            initial_box = [i for i in sorted(enumerate(points_sum), key=lambda x:x[1][0])][0]
            print(initial_box)
            x_min = initial_box[1][1][0]
            x_max = initial_box[1][2][0]
            threshold = abs(x_max - x_min) / 2 + 5
            new_sorted.append(initial_box)
            del points_sum[initial_box[0]]

            # find boxes that are in same col with initial box, below the threshold
            same_col = list(map(lambda x:[abs(x[1][0]-x_min),x[1],x[2]],points_sum))
            same_col = [[count,i] for count,i in enumerate(same_col)]
            same_col = [i for i in same_col if i[1][0] <= threshold]

            # sort boxes in the same col based on ymin
            sorted_same_col = list(map(lambda x:[x[0],x[1]],sorted(same_col,key=lambda x:x[1][2][1])))

            # remove sorted boxes
            del_indx = []
            for i in sorted_same_col:
                new_sorted.append(i)
                del_indx.append(i[0])

            for id in sorted(del_indx,reverse=True):
                del points_sum[id]

            final_sorted_list.append(new_sorted)
        except Exception as e:
            print(e)
            print(points_sum)
            break

    # sigh row and col
    print(len(final_sorted_list))
    out_put = []
    start_point = start_point.split("_")
    start_col = int(start_point[0])
    start_row = int(start_point[1])
    print(f"start point is {start_point},start col is {start_col}, start row it {start_row}")
    for i in range(len(final_sorted_list)):
        col = start_col - i
        for j in range(len(final_sorted_list[i])):
            if col % 2 == 0:
                # row = 80 - start_row + 1 - j
                row = start_row + j
            else:
                row = start_row + j
            print(final_sorted_list[i][j])
            out_put.append(dict(
                label = str(col)+"_"+str(row),
                points = [final_sorted_list[i][j][1][1],final_sorted_list[i][j][1][2]],
                shape_type = "rectangle"
            ))
    print(out_put)

    data["shapes"]= out_put
    a = file.replace(".json","_sorted.json")
    with open(file, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    # file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/label/DJI_00001.json"
    # sortbox(file,start_point="5_1")
    # file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/label/DJI_00002.json"
    # sortbox(file,start_point="5_2")
    # file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/label/DJI_00003.json"
    # sortbox(file,start_point="5_7")
    # file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/label/DJI_00004.json"
    # sortbox(file,start_point="6_12")
    # file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/label/DJI_00005.json"
    # sortbox(file,start_point="6_16")
    # file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/label/DJI_00006.json"
    # sortbox(file,start_point="6_21")
    # file = "../../../../../result/yolov3_train_val/PCFS_nursery/label/DJI_00007.json"
    # sortbox(file,start_point="6_27")
    start_list =["5_1","5_2","5_7","6_12","6_16","6_21","6_27","6_32","6_37","6_43","6_47","6_53","6_57","6_63","6_67","6_73",
                 "10_72","10_62",
                 ]
    file = "../../../../../result/yolov3_train_val/PCFS_nursery/label/DJI_00020.json"
    sortbox(file,start_point="10_57")
