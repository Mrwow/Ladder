import numpy as np
import os
import json
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import random


def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

class ConfusionMatrix:
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.25, IOU_THRESHOLD=0.45):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)
        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]
        print(all_matches)

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                print(i)
                print(label)
                self.matrix[self.num_classes, gt_class] += 1

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or ( all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0 ):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

def confusionMatrix(label_fl, prediction_fl):
    label = []
    with open(label_fl, "r") as f:
        data = json.load(f)
    for shape in data["shapes"]:
        box = [int(shape["label"]),
               shape["points"][0][0],shape["points"][0][1],
               shape["points"][1][0],shape["points"][1][1],
               ]
        label.append(box)

    predict = []
    with open(prediction_fl, "r") as f:
        data = json.load(f)
    for shape in data["shapes"]:
        box = [shape["points"][0][0],shape["points"][0][1],
               shape["points"][1][0],shape["points"][1][1],
               shape['score'],
               int(float(shape["label"]))
               ]
        predict.append(box)

    predict = np.array(predict)
    label = np.array(label)
    confusion_matrix = ConfusionMatrix(num_classes = 3)
    confusion_matrix.process_batch(predict, label)
    a = confusion_matrix.return_matrix()
    print("++++++")
    print(label_fl)
    print(a)
    print("++++++")
    return a

def confusionMatrixMultiImg(truth_fd,pre_fd):
    cm_all = np.zeros((3 + 1, 3 + 1))
    for f in os.listdir(truth_fd):
        if f.endswith('.json'):
            json_file_truth = os.path.join(truth_fd,f)
            json_file_pre = os.path.join(pre_fd,f)
            cm = confusionMatrix(json_file_truth,json_file_pre)
            # print(cm)
            cm_all += cm
    print("final")
    print(cm_all)
    return cm_all


def plot(matrix, nc, normalize=True, save_dir='', names=()):
    try:
        import seaborn as sn

        array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < len(names) < 99) and len(names) == nc  # apply names to ticklabels
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array, annot=nc < 30, annot_kws={"size": 20}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
        fig.axes[0].set_xlabel('True')
        fig.axes[0].set_ylabel('Predicted')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close()
    except Exception as e:
        print(f'WARNING: ConfusionMatrix plot failure: {e}')


if __name__ == '__main__':
    label_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/metrics/ground_truth"
    # prediction_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/crop_test/impute"
    # label_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/crop_test/YOLO"
    # prediction_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/crop_test/YOLO"
    prediction_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/crop_test/prediction_onlyYOLO"
    # confusionMatrix(label_fd,prediction_fd)
    cm_all = confusionMatrixMultiImg(label_fd,prediction_fd)
    # predict_fl = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/metrics/exp18/exp83_f1train_best_predictions.json"
    # confusionMatrix_2(label_fd,predict_fl)
    # confusionMatrix_3(prediction_fd,predict_fl)
    plot(matrix=cm_all,nc=3,save_dir=prediction_fd, names=["Light","Moderate","High"])