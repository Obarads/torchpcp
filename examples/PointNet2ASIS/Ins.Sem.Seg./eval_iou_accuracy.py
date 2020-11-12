import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../../../"))) # for package path

import numpy as np
from scipy import stats

import argparse
from tqdm import tqdm

NUM_CLASSES = 13

parser = argparse.ArgumentParser()
parser.add_argument("--output_filelist", "-f", required=True)
parser.add_argument("--result_file", "-r", required=True)
args = parser.parse_args()

pred_data_label_filenames = []
# for i in range(5,6):
#   file_name = 'log{}_test/output_filelist.txt'.format(i)
#   pred_data_label_filenames += [line.rstrip() for line in open(file_name)]

file_name = args.output_filelist
folder = os.path.dirname(file_name)
oscwd = os.getcwd()
pred_data_label_filenames += [os.path.join(oscwd, folder, line.rstrip()) for line in open(file_name)]

gt_label_filenames = [os.path.join(oscwd, folder, f.rstrip('_pred\.txt')) + '_gt.txt' for f in pred_data_label_filenames]

num_room = len(gt_label_filenames)

# Initialize...
# acc and macc
total_true = 0
total_seen = 0
true_positive_classes = np.zeros(NUM_CLASSES)
positive_classes = np.zeros(NUM_CLASSES)
gt_classes = np.zeros(NUM_CLASSES)
# mIoU
ious = np.zeros(NUM_CLASSES)
totalnums = np.zeros(NUM_CLASSES)
# precision & recall
total_gt_ins = np.zeros(NUM_CLASSES)
at = 0.5
tpsins = [[] for itmp in range(NUM_CLASSES)]
fpsins = [[] for itmp in range(NUM_CLASSES)]
# mucov and mwcov
all_mean_cov = [[] for itmp in range(NUM_CLASSES)]
all_mean_weighted_cov = [[] for itmp in range(NUM_CLASSES)]


room_indexes = tqdm(range(num_room), desc="eval ins.&sem.seg.", ncols=70)

for i in room_indexes:
    # print(i)
    data_label = np.loadtxt(pred_data_label_filenames[i])
    pred_ins = data_label[:,-1].reshape(-1).astype(np.int)
    pred_sem = data_label[:, -2].reshape(-1).astype(np.int)
    gt_label = np.loadtxt(gt_label_filenames[i])
    gt_ins = gt_label[:, -1].reshape(-1).astype(np.int)
    gt_sem = gt_label[:, -2].reshape(-1).astype(np.int)
    # print(gt_label.shape)

    # semantic acc
    total_true += np.sum(pred_sem == gt_sem)
    total_seen += pred_sem.shape[0]

    # pn semantic mIoU
    for j in range(gt_sem.shape[0]):
        gt_l = int(gt_sem[j])
        pred_l = int(pred_sem[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l==pred_l)

    # semantic mIoU
    #intersections = np.zeros(NUM_CLASSES)
    #unions = np.zeros(NUM_CLASSES)
    #un, indices = np.unique(gt_sem, return_index=True)
    #for segid in un:
    #    intersect = np.sum((pred_sem == segid) & (gt_sem == segid))
    #    union = np.sum((pred_sem == segid) | (gt_sem == segid))
    #    intersections[segid] += intersect
    #    unions[segid] += union
    #    true_positive_classes[segid] += intersect
    #    gt_classes[segid] += np.sum(gt_sem == segid)

    #iou = intersections / unions
    #for i_iou, iou_ in enumerate(iou):
    #    if not np.isnan(iou_):
    #        ious[i_iou] += iou_
    #        totalnums[i_iou] += 1

    # instance
    un = np.unique(pred_ins)
    pts_in_pred = [[] for itmp in range(NUM_CLASSES)]
    for ig, g in enumerate(un):  # each object in prediction
        if g == -1:
            continue
        tmp = (pred_ins == g)
        sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
        pts_in_pred[sem_seg_i] += [tmp]

    un = np.unique(gt_ins)
    pts_in_gt = [[] for itmp in range(NUM_CLASSES)]
    for ig, g in enumerate(un):
        tmp = (gt_ins == g)
        sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
        pts_in_gt[sem_seg_i] += [tmp]

    # instance mucov & mwcov
    for i_sem in range(NUM_CLASSES):
        sum_cov = 0
        mean_cov = 0
        mean_weighted_cov = 0
        num_gt_point = 0
        for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
            ovmax = 0.
            num_ins_gt_point = np.sum(ins_gt)
            num_gt_point += num_ins_gt_point
            for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                union = (ins_pred | ins_gt)
                intersect = (ins_pred & ins_gt)
                iou = float(np.sum(intersect)) / np.sum(union)

                if iou > ovmax:
                    ovmax = iou
                    ipmax = ip

            sum_cov += ovmax
            mean_weighted_cov += ovmax * num_ins_gt_point

        if len(pts_in_gt[i_sem]) != 0:
            mean_cov = sum_cov / len(pts_in_gt[i_sem])
            all_mean_cov[i_sem].append(mean_cov)

            mean_weighted_cov /= num_gt_point
            all_mean_weighted_cov[i_sem].append(mean_weighted_cov)


    # instance precision & recall
    for i_sem in range(NUM_CLASSES):
        tp = [0.] * len(pts_in_pred[i_sem])
        fp = [0.] * len(pts_in_pred[i_sem])
        gtflag = np.zeros(len(pts_in_gt[i_sem]))
        total_gt_ins[i_sem] += len(pts_in_gt[i_sem])

        for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
            ovmax = -1.

            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                union = (ins_pred | ins_gt)
                intersect = (ins_pred & ins_gt)
                iou = float(np.sum(intersect)) / np.sum(union)

                if iou > ovmax:
                    ovmax = iou
                    igmax = ig

            if ovmax >= at:
                    tp[ip] = 1  # true
            else:
                fp[ip] = 1  # false positive

        tpsins[i_sem] += tp
        fpsins[i_sem] += fp


MUCov = np.zeros(NUM_CLASSES)
MWCov = np.zeros(NUM_CLASSES)
for i_sem in range(NUM_CLASSES):
    MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
    MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])

precision = np.zeros(NUM_CLASSES)
recall = np.zeros(NUM_CLASSES)
for i_sem in range(NUM_CLASSES):
    tp = np.asarray(tpsins[i_sem]).astype(np.float)
    fp = np.asarray(fpsins[i_sem]).astype(np.float)
    tp = np.sum(tp)
    fp = np.sum(fp)
    rec = tp / total_gt_ins[i_sem]
    prec = tp / (tp + fp)

    precision[i_sem] = prec
    recall[i_sem] = rec


LOG_FOUT = open(os.path.join('results_a5.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)



# instance results
log_string('Instance Segmentation MUCov: {}'.format(MUCov))
log_string('Instance Segmentation mMUCov: {}'.format(np.mean(MUCov)))
log_string('Instance Segmentation MWCov: {}'.format(MWCov))
log_string('Instance Segmentation mMWCov: {}'.format(np.mean(MWCov)))
log_string('Instance Segmentation Precision: {}'.format(precision))
log_string('Instance Segmentation mPrecision: {}'.format(np.mean(precision)))
log_string('Instance Segmentation Recall: {}'.format(recall))
log_string('Instance Segmentation mRecall: {}'.format(np.mean(recall)))

# semantic results
iou_list = []
for i in range(NUM_CLASSES):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
    print(iou)
    iou_list.append(iou)

log_string('Semantic Segmentation oAcc: {}'.format(sum(true_positive_classes)/float(sum(positive_classes))))
#log_string('Semantic Segmentation Acc: {}'.format(true_positive_classes / gt_classes))
log_string('Semantic Segmentation mAcc: {}'.format(np.mean(true_positive_classes / gt_classes)))
log_string('Semantic Segmentation IoU: {}'.format(iou_list))
log_string('Semantic Segmentation mIoU: {}'.format(1.*sum(iou_list)/NUM_CLASSES))

from torch_point_cloud.datasets.ASIS.utils.indoor3d_util import g_classes
from torch_point_cloud.utils.converter import dict2mdtables

def digit_format(data_list):
    return np.round(data_list, 4)

expand_g_classes = ["metrics","mean", *g_classes]

tables = {
    "Instance Segmentation":{
        "head": expand_g_classes,
        "body": [
            ["MUCov", np.mean(MUCov), *digit_format(MUCov)],
            ["MWCov", np.mean(MWCov), *digit_format(MWCov)],
            ["Precision", np.mean(precision), *digit_format(precision)],
            ["Recall", np.mean(recall), *digit_format(recall)]
        ]
    },
    "Semantic Segmentation": {
        "head": expand_g_classes,
        "body": [
            ["oAcc", sum(true_positive_classes)/float(sum(positive_classes))],
            ["mAcc", np.mean(true_positive_classes / gt_classes)],
            ["IoU", 1.*sum(iou_list)/NUM_CLASSES, *digit_format(iou_list)]
        ]
    }
}

dict2mdtables(tables, args.result_file)
print("save results:", args.result_file)
