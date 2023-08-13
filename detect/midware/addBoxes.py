import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def addBox(shapes,im0_w=5472, im0_h=3648):
    if isinstance(shapes,str):
        with open(shapes, "r") as f:
            data = json.load(f)

        shapes = [
            dict(
                label=s["label"],
                points=s["points"]
            )
            for s in data["shapes"]
        ]

    # get the boxes size distribution
    points = [s["points"] for s in shapes]
    # print(len(points))
    widths = []
    heights = []
    centers = []
    xs = []
    ys = []
    for box in points:
        top_left = box[0]
        bot_right = box[1]
        center_x = (top_left[0] + bot_right[0])/2
        center_y = (top_left[1] + bot_right[1])/2
        w = bot_right[0] - top_left[0]
        h = bot_right[1] - top_left[1]
        # print(f'x:{center_x}, y:{center_y}, width: {w}, height:{h}')
        widths.append(w)
        heights.append(h)
        centers.append([center_x,im0_h-center_y])
        xs.append(center_x)
        ys.append(im0_h - center_y)
        # ys.append(center_y)

    avg_w = sum(widths)/len(widths)
    avg_h = sum(heights)/len(widths)
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.scatter(xs,ys,alpha=0.2, color="gray",)

    # filter points
    xs_filter = []
    ys_filter = []
    centers_filter = []
    widths_filter = []
    heights_filter = []
    filter_index = []
    for i, center in enumerate(centers):
        if 0.6 * avg_w  < widths[i] < 1.5 * avg_w and \
                0.6 * avg_h < heights[i] < 1.5 * avg_h :
            xs_filter.append(xs[i])
            ys_filter.append(ys[i])
            centers_filter.append(centers[i])
            widths_filter.append(widths[i])
            heights_filter.append(heights[i])
        else:
            filter_index.append(i)
    ax2.scatter(xs_filter,ys_filter,alpha=0.2, color="red",)

    # fill points
    final_sorted_list_col = []
    # print(len(centers_filter))
    centers_filter_sorted_col = sorted(centers_filter,key=lambda x:x[0])
    centers_filter_sorted_row = sorted(centers_filter,key=lambda x:x[1])
    y_min = centers_filter_sorted_row[0][1]
    y_max = centers_filter_sorted_row[-1][1]
    # print(f'min y is {y_min}, max y is {y_max}')
    # print(centers_filter_sorted_col )
    # print(f'avg_w is {avg_w}, avg_h is {avg_h}')
    # print("===============")
    while True:
        try:
            # find the initial box
            initial = centers_filter_sorted_col [0]
            same_col =[]
            same_col.append(initial)
            del centers_filter_sorted_col [0]

            # find points that are in same col with initial box, below the threshold
            del_indx = []
            for i, center in enumerate(centers_filter_sorted_col ):
                if abs(center[0] - initial[0]) < (0.6 * avg_w):
                    same_col.append(center)
                    del_indx.append(i)
                    # print(same_col)

            # sort boxes in the same col based on y
            sorted_same_col = sorted(same_col,key=lambda x:x[1])
            final_sorted_list_col.append(sorted_same_col)
            # print(sorted_same_col)

            for j in sorted(del_indx,reverse=True):
                del centers_filter_sorted_col [j]

        except Exception as e:
            print(e)
            break

    out = [len(l) for l in final_sorted_list_col]
    print(out)
    # print(final_sorted_list_col[2])

    ax3.scatter(xs_filter,ys_filter,alpha=0.2, color="red",)
    new_l = []
    for l in final_sorted_list_col:
        # fill between the 1st and ymin
        if l[0][1] > y_min:
            x1 = l[0][0]
            y1 = l[0][1]
            x2 = x1
            y2 = y_min
            if y1-y2 > 1.1 * avg_h:
                n = int(abs(y1-y2)/avg_h)
                x_ = (x1+x2)/2
                for m in range(n):
                    y_ = y1 - avg_h * (m+1) * 1.1
                    ax3.add_patch(Rectangle((x_- avg_w/2, y_ - avg_h/2),avg_w,avg_h))
                    ax3.scatter(x_,y_,alpha=0.6, color="yellow",)
                    new_l.append(center2point([x_, y_],avg_w,avg_h, im0_w, im0_h))

        # fill between points
        for k in range(len(l)):
            y1 = l[k][1]
            x1 = l[k][0]

            if k < len(l) -1 :
                y2 = l[k+1][1]
                x2 = l[k+1][0]

                if y2-y1 > 1.2 * avg_h:
                    n = int((y2-y1)/avg_h)
                    x_ = (x1+x2)/2
                    for m in range(n-1):
                        y_ = y1 + avg_h * (m+1) * 1.1
                        ax3.add_patch(Rectangle((x_- avg_w/2, y_ - avg_h/2),avg_w,avg_h))
                        ax3.scatter(x_,y_,alpha=0.6, color="yellow",)
                        new_l.append(center2point([x_, y_],avg_w,avg_h,im0_w,im0_h))

            if k == len(l) -1:
                x2 = x1
                y2 = y_max

                if y2-y1 > 1.1 * avg_h:
                    n = int((y2-y1)/avg_h)
                    x_ = (x1+x2)/2
                    for m in range(n):
                        y_ = y1 + avg_h * (m+1) * 1.1
                        ax3.add_patch(Rectangle((x_- avg_w/2, y_ - avg_h/2),avg_w,avg_h))
                        ax3.scatter(x_,y_,alpha=0.6, color="yellow",)
                        new_l.append(center2point([x_, y_],avg_w,avg_h, im0_w,im0_h))
    # plt.show()
    print(new_l)

    return new_l, filter_index

def center2point(center,box_w,box_h,im0_w,im0_h):
    x1 = center[0] - box_w/2
    x1 = max(0,x1)
    x2 = center[0] + box_w/2
    x2 = min(x2,im0_w)
    y1 = center[1] + box_h/2
    y2 = center[1] - box_h/2
    return (x1,im0_h-y1,x2,im0_h-y2)


if __name__ == '__main__':
    file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/addBox/DJI_00003.json"
    # file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/addBox/DJI_00040.json"
    # file = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/addBox/DJI_00018.json"
    addBox(file)
