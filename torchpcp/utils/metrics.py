import torch
import numpy as np
from scipy.stats import mode

class LossMeter:
    """
    Mean loss.
    (warning: this class does not consider batch size. compute function divide sum losses by number of times the update function was called.)
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss_list = []

    def update(self, losses):
        """
        update losses.

        Parameters
        ----------
        losses: list or numpy.array
            losses.shape=()
        """
        self.loss_list.append(losses)

    def compute(self):
        res = np.sum(self.loss_list) / len(self.loss_list)
        return res.tolist()


"""
Original code by : mit-han-lab
https://github.com/mit-han-lab/pvcnn/blob/master/meters/s3dis.py
"""
class AssessmentMeter:
    def __init__(self, num_classes, metric='iou'):
        super().__init__()
        assert metric in ['overall', 'class', 'iou']
        self.metric = metric
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total_seen = [0] * self.num_classes
        self.total_correct = [0] * self.num_classes
        self.total_positive = [0] * self.num_classes
        self.total_seen_num = 0
        self.total_correct_num = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        args:
            outputs: B x 13 x num_points

            targets: B x num_points
        """
        predictions = outputs.argmax(1)
        if self.metric == 'overall':
            self.total_seen_num += targets.numel()
            self.total_correct_num += torch.sum(targets == predictions).item()
        else:
            # self.metric == 'class' or self.metric == 'iou':
            for i in range(self.num_classes):
                itargets = (targets == i)
                ipredictions = (predictions == i)
                self.total_seen[i] += torch.sum(itargets).item()
                self.total_positive[i] += torch.sum(ipredictions).item()
                self.total_correct[i] += torch.sum(itargets & ipredictions).item()

    def compute(self):
        if self.metric == 'class':
            accuracy = 0
            for i in range(self.num_classes):
                total_seen = self.total_seen[i]
                if total_seen == 0:
                    accuracy += 1
                else:
                    accuracy += self.total_correct[i] / total_seen
            return accuracy / self.num_classes
        elif self.metric == 'iou':
            iou = 0
            for i in range(self.num_classes):
                total_seen = self.total_seen[i]
                if total_seen == 0:
                    iou += 1
                else:
                    total_correct = self.total_correct[i]
                    iou += total_correct / (total_seen + self.total_positive[i] - total_correct)
            return iou / self.num_classes
        else:
            return self.total_correct_num / self.total_seen_num

class MultiAssessmentMeter:
    def __init__(self, num_classes, metrics=["class", "iou", "overall"]):
        for metric in metrics: assert metric in ["class", "iou", "overall"]

        meters = {}
        for metric in metrics:
            meters[metric]=AssessmentMeter(num_classes=num_classes, metric=metric)

        self.meters = meters
        self.metrics = metrics
    
    def reset(self):
        for key in self.meters:
            self.meters[key].reset()

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        for key in self.meters:
            self.meters[key].update(outputs, targets)
        
    def compute(self):
        results = {}
        for key in self.meters:
            results[key] = self.meters[key].compute()
        
        return results

class InsMeter:
    def __init__(self, num_classes, metric='iou'):
        assert metric in ['overall', 'class', 'iou']
        self.metric = metric
        self.num_classes = num_classes
        # self.size_threshold = 0.25 * sizes
        self.size_threshold = 0.25 * np.ones((num_classes))
        self.reset()

    def reset(self):
        self.total = np.zeros(self.num_classes)
        self.fps   = [[] for i in range(self.num_classes)]
        self.tps   = [[] for i in range(self.num_classes)]

    def update(self, pred_ins_labels: torch.Tensor, gt_ins_labels: torch.Tensor,
               pred_sem_labels: torch.Tensor, gt_sem_labels: torch.Tensor):
        for idx in range(len(pred_ins_labels)):
            pil = pred_ins_labels[idx]
            gil = gt_ins_labels[idx]
            psl = pred_sem_labels[idx]
            gsl = gt_sem_labels[idx]

            # pred
            proposals = [[] for i in range(self.num_classes)]
            for gid in np.unique(pil):
                indices = (pil[:] == gid)
                sem = int(mode(psl[indices])[0])
                size = np.sum(indices)
                if size > self.size_threshold:
                    proposals[sem] += [indices]

            # gt
            instances = [[] for i in range(self.num_classes)]
            for gid in np.unique(gil):
                indices = (gil[:] == gid)
                sem = int(mode(gsl[indices])[0])
                instances[sem] += [indices]

            for i in range(self.num_classes):
                self.total[i] += len(instances[i])
                tp = np.zeros(len(proposals[i]))
                fp = np.zeros(len(proposals[i]))
                gt = np.zeros(len(instances[i]))
                for pid, u in enumerate(proposals[i]):
                    overlap = 0.0
                    detected = 0
                    for iid, v in enumerate(instances[i]):
                        iou = np.sum((u & v)) / np.sum((u | v))
                        if iou > overlap:
                            overlap = iou
                            detected = iid
                    if overlap >= 0.5:
                        tp[pid] = 1
                    else:
                        fp[pid] = 1
                self.tps[i] += [tp]
                self.fps[i] += [fp]

