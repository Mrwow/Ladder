import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cofusionMatrix import box_iou_calc
import os
import json
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc

def ApPerClassMultiImg(label_fd, prediction_fd):
    correct,pre_score,pre_cls,true_cls = [],[],[],[]
    for f in os.listdir(label_fd):
        if f.endswith('.json'):
            json_file_truth = os.path.join(label_fd,f)
            json_file_pre = os.path.join(prediction_fd,f)
            correct_,pre_score_,pre_cls_,true_cls_ =  ApPerClassOneImage(json_file_truth, json_file_pre)
            correct.append(correct_)
            pre_score.append(pre_score_)
            pre_cls.append(pre_cls_)
            true_cls.append(true_cls_)
    correct = np.concatenate(correct,0)
    pre_score = np.concatenate(pre_score,0)
    pre_cls = np.concatenate(pre_cls,0)
    true_cls = np.concatenate(true_cls,0)
    print(correct.shape)
    names = {0: 'Low', 1: 'Moderate', 2: 'High'}
    p, r, ap, f1, ap_class = ap_per_class(correct,pre_score,pre_cls,true_cls, plot=True, save_dir=prediction_fd, names=names)

def ApPerClassOneImage(label_fl, prediction_fl):
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
    correct = process_batch(predict,label)

    return correct,predict[:,4],predict[:,5],label[:,0]

def process_batch(detections, labels, iou_thred=0.45):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """

    # iou = box_iou(labels[:, 1:], detections[:, :4])
    # correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iouv=np.linspace(0.5,0.95,10)
    correct = np.zeros((detections.shape[0], iouv.shape[0]))
    iou = box_iou_calc(labels[:, 1:], detections[:, :4])
    x = np.where((iou >= iou_thred) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = [[x[0][i], x[1][i], iou[x[0][i], x[1][i]]]
                   for i in range(x[0].shape[0])]
        matches =np.array(matches)
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        s = matches[:, 2:3] >= iouv
        for i in range(s.shape[0]):
            correct[int(matches[i][1])] = s[i]
        correct = correct.astype(np.bool)
    return correct

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    recall_for_plot = []
    precision_for_plot =[]
    FDR_for_plot = []
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, px because px decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
            fdr = fpc/(tpc + fpc)

            # AP from recall-precision curve at each IOU level
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
                    recall_for_plot.append(recall[:, j])
                    precision_for_plot.append(precision[:, j])
                    FDR_for_plot.append(fdr[:,j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_raw_pr_curve(recall_for_plot,precision_for_plot,Path(save_dir) / 'PR_curve_raw.png', names)
        plot_raw_fdr(FDR_for_plot,recall_for_plot,Path(save_dir) / 'FDR_curve_raw.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

def plot_raw_pr_curve(recall,precision,save_dir,names):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    for i in range(len(names)):

        mrec = np.concatenate(([0.0],recall[i], [1.0]))
        mpre = np.concatenate(([1.0],precision[i], [0.0]))
        ap = auc(mrec,mpre)
        ax.plot(mrec,mpre, linewidth=3, label=f'{names[i]} {ap:.3f}')

    ax.set_xlabel('Recall',fontsize = 20)
    ax.set_ylabel('Precision', fontsize = 20)
    plt.legend(fontsize=18)
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

def plot_raw_fdr(fdr,recall,save_dir,names):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    for i in range(len(names)):
        ax.plot(fdr[i], recall[i], linewidth=1, label=f'{names[i]} ')

    ax.set_xlabel('FDR')
    ax.set_ylabel('Recall')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre))) ##

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

# Plots ----------------------------------------------------------------------------------------------------------------
def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

if __name__ == '__main__':
    # label_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/metrics/ground_truth/"
    label_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov5/72-64_image/val_spillmanWin_conf0.3_iou0.5/ground_truth"
    prediction_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov5/72-64_image/val_spillmanWin_conf0.3_iou0.5/img"

    # prediction_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/crop_test/impute"
    # prediction_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov3_train_val/PCFS_nursery/crop_test/prediction_onlyYOLO"
    # prediction_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov5_resnet/train_f1_clean_aug_clean+resnet/prediction_yolo5_del"
    label_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov5_resnet/train_f1_clean_aug_clean+resnet/ground_truth"
    prediction_fd = "/Users/ZhouTang/Downloads/zzlab/1_Project/Wheat_rust_severity/source/result/yolov5_resnet/train_f1_clean_aug_clean+resnet/prediction_yolo5_imputed&del"
    ApPerClassMultiImg(label_fd,prediction_fd)