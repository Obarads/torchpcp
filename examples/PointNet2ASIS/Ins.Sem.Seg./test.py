import os, sys
CW_DIR = os.getcwd()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "configs/test.yaml"))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../../../"))) # for package path

import hydra
import omegaconf
import numpy as np
from tqdm import tqdm
from scipy import stats

import torch

# dataset
from torch_point_cloud.datasets.ASIS.utils.test_utils import BlockMerging
from torch_point_cloud.datasets.ASIS.S3DIS import create_batch_instance_information

# tools
from torch_point_cloud.utils.setting import (PytorchTools, get_configs,
                                             make_folders, fix_path_in_configs)
from torch_point_cloud.utils.metrics import MultiAssessmentMeter, LossMeter
from torch_point_cloud.utils import converter

# env
from model_env import processing, cluster
from model_env import get_model, get_dataset, get_losses, get_checkpoint

@hydra.main(config_name=CONFIG_PATH)
def main(cfg : omegaconf.DictConfig):
    # fix paths
    cfg = fix_path_in_configs(
        CW_DIR,
        cfg,
        [
            ["dataset","root"],
            ["general", "sizetxt"],
            ["model", "resume"]
        ]
    )

    # set a seed 
    PytorchTools.set_seed(
        cfg.general.seed, 
        cfg.general.device, 
        cfg.general.reproducibility
    )

    # set a device
    cfg.general.device = PytorchTools.select_device(cfg.general.device)

    # get a trained model env
    checkpoint, checkpoint_cfg = get_checkpoint(cfg.model.resume)

    # change cfg
    checkpoint_cfg.dataset.root = cfg.dataset.root
    checkpoint_cfg.dataset.name = "s3dis"
    checkpoint_cfg.dataset.mode = "scene"

    model = get_model(checkpoint_cfg)
    dataset = get_dataset(checkpoint_cfg)
    criterion = get_losses(checkpoint_cfg)

    # set trained params
    model.load_state_dict(checkpoint["model"])

    # get ins size txt file
    mean_num_pts_in_group = np.loadtxt(cfg.general.sizetxt)

    # test
    eval(model, dataset["test"], criterion, mean_num_pts_in_group, 
         cfg.general.device)

def create_processing_data(point_clouds, gt_sem_labels, gt_ins_labels):
    """
    create data of progress function args.
    Parameters
    ----------
    block_point_clouds: np.array (shape: (B, N, C))
    block_gt_sem_labels: np.array (shape: (B, N))
    block_gt_ins_labels: np.array (shape: (B, N))
    """

    new_gt_ins_labels, gt_ins_masks, gt_ins_label_sizes, \
        maximum_num_instance = create_batch_instance_information(gt_ins_labels)
    new_gt_ins_labels = torch.tensor(new_gt_ins_labels)
    gt_ins_masks = torch.tensor(gt_ins_masks)
    gt_ins_label_sizes = torch.tensor(gt_ins_label_sizes)

    point_clouds = torch.tensor(point_clouds)
    gt_sem_labels = torch.tensor(gt_sem_labels)
    data = (point_clouds, gt_sem_labels, new_gt_ins_labels, gt_ins_label_sizes,
            gt_ins_masks)

    return data

# evaluation
def eval(
    model, 
    dataset, 
    criterion, 
    mean_num_pts_in_group, 
    device,
    save_results=True
):
    model.eval()

    total_acc = 0.0
    total_seen = 0

    output_filelist_f = 'output_filelist.txt'
    fout_out_filelist = open(output_filelist_f, 'w')

    semantic_meters = MultiAssessmentMeter(
        num_classes=dataset.num_classes, 
        metrics=["class","overall","iou"]
    )
    batch_loss = LossMeter()
    meters = (semantic_meters, batch_loss)

    scene_idx_list = range(len(dataset))
    scene_idx_list = tqdm(scene_idx_list, ncols=80, desc="test")

    # dataset = secne dataset
    for scene_idx in scene_idx_list:
        # scene_idx = 3
        scene = dataset[scene_idx]
        block_idx_list = range(len(scene[0]))

        scene_point_cloud = scene[0]
        scene_gt_sem_label = scene[1]
        scene_gt_ins_label = scene[2]
        scene_name = scene[3]
        raw_scene_data = scene[4]

        pred_file_name = scene_name+"_pred.txt"
        gt_file_name = scene_name+"_gt.txt"
        fout_data_label = open(pred_file_name, 'w')
        fout_gt_label = open(gt_file_name, 'w')

        fout_out_filelist.write(pred_file_name+"\n")

        max_scene_x = max(raw_scene_data[:,0])
        max_scene_y = max(raw_scene_data[:,1])
        max_scene_z = max(raw_scene_data[:,2])

        block_pred_sem_label_list = np.zeros_like(scene_gt_sem_label)
        block_pred_sem_output_list = np.zeros([scene_gt_sem_label.shape[0], 
                                                scene_gt_sem_label.shape[1], 
                                                dataset.num_classes])
        block_pred_ins_label_list = np.zeros_like(scene_gt_ins_label)

        # make_folders(scene_name)

        gap = 5e-3
        volume_num = int(1. / gap)+1
        volume = -1* np.ones([volume_num,volume_num,volume_num]).astype(np.int32)
        volume_seg = -1* np.ones([volume_num,volume_num,volume_num]).astype(np.int32)

        for block_idx in block_idx_list:
            # scene[0]: point clouds
            # scene[1]: semantic labels
            # scene[2]: instance labels
            # The reason for using array slice is that the model can only 
            # processe batch data.
            # shape: (B, N, -1) (Batch size is 1.)
            block_point_clouds = scene_point_cloud[block_idx:block_idx+1]
            block_gt_sem_labels = scene_gt_sem_label[block_idx:block_idx+1]
            block_gt_ins_labels = scene_gt_ins_label[block_idx:block_idx+1]

            # create data 
            data = create_processing_data(block_point_clouds, 
                                          block_gt_sem_labels, 
                                          block_gt_ins_labels)

            # model processing
            with torch.no_grad():
                loss, block_pred_sem_outputs, block_ins_embeddings = processing(
                    model, criterion, data, meters, device, return_outputs=True)

            # block losses
            loss = PytorchTools.t2n(loss)

            # get data of block from data of blocks
            # (B, N, -1) -> (N, -1)  (Batch size is 1.)
            block_pred_sem_output = PytorchTools.t2n(block_pred_sem_outputs[0].transpose(0,1)) # for test: comment out
            block_pred_sem_output_list[block_idx] = block_pred_sem_output

            # block_pred_sem_label = block_gt_sem_labels[0].astype(np.float32) # for test
            blcok_ins_embedding = PytorchTools.t2n(block_ins_embeddings[0])
            block_point_cloud = block_point_clouds[0]
            block_gt_sem_label = block_gt_sem_labels[0]
            block_gt_ins_label = block_gt_ins_labels[0]

            # semantic possibility to semantic labels
            block_pred_sem_label = np.argmax(block_pred_sem_output, axis=1) # for test: comment out
            block_pred_sem_label_list[block_idx, :] = block_pred_sem_label

            # cluster (for instance segmentation)
            # get prediction instance labels and cluster data
            ins_seg = {}
            num_clusters, block_pred_ins_label, cluster_centers = cluster(
                 blcok_ins_embedding, bandwidth=0.6) # for test: comment out
            # block_pred_ins_label = PytorchTools.t2n(data[2][0]) # for test
            # num_clusters = len(np.unique(block_pred_ins_label)) # for test

            # print(block_pred_ins_label.shape,block_pred_ins_label)
            # print(np.unique(block_pred_ins_label))

            for idx_cluster in range(num_clusters):
                tmp = (block_pred_ins_label == idx_cluster)
                if np.sum(tmp) != 0: # add (for a cluster of zero element.)
                    a = stats.mode(block_pred_sem_label[tmp])[0]
                    estimated_seg = int(a)
                    ins_seg[idx_cluster] = estimated_seg

            # I should change this value name
            merged_block_pred_ins_label = BlockMerging(volume, volume_seg, 
                                                       block_point_cloud[:, 6:],
                                                       block_pred_ins_label, 
                                                       ins_seg, gap)

            # labels2ply(os.path.join(scene_name, "block_{}m.ply".format(block_idx)), 
            #            block_point_cloud, merged_block_pred_ins_label.astype(np.int32), 
            #            seed=0)

            block_pred_ins_label_list[block_idx, :] = merged_block_pred_ins_label
            total_acc += float(np.sum(block_pred_sem_label==block_gt_sem_label)) \
                / block_pred_sem_label.shape[0]
            total_seen += 1

        # from blocks to a scene (B, N, -1) -> (B*N, -1) (B is block size in a scene)
        block_pred_ins_label_list = block_pred_ins_label_list.reshape(-1)
        block_pred_sem_label_list = block_pred_sem_label_list.reshape(-1)
        block_pred_sem_output_list = block_pred_sem_output_list.reshape([-1, dataset.num_classes])
        scene_point_cloud_from_blocks = scene_point_cloud.reshape([-1, 9])

        # filtering
        x = (scene_point_cloud_from_blocks[:, 6] / gap).astype(np.int32)
        y = (scene_point_cloud_from_blocks[:, 7] / gap).astype(np.int32)
        z = (scene_point_cloud_from_blocks[:, 8] / gap).astype(np.int32)
        for i in range(block_pred_ins_label_list.shape[0]):
            if volume[x[i], y[i], z[i]] != -1:
                block_pred_ins_label_list[i] = volume[x[i], y[i], z[i]]

        un = np.unique(block_pred_ins_label_list)
        pts_in_pred = [[] for itmp in range(dataset.num_classes)]
        group_pred_final = -1 * np.ones_like(block_pred_ins_label_list)
        grouppred_cnt = 0
        for ig, g in enumerate(un): #each object in prediction
            if g == -1:
                continue
            tmp = (block_pred_ins_label_list == g)
            sem_seg_g = int(stats.mode(block_pred_sem_label_list[tmp])[0])
            #if np.sum(tmp) > 500:
            if np.sum(tmp) > 0.25 * mean_num_pts_in_group[sem_seg_g]:
                group_pred_final[tmp] = grouppred_cnt
                pts_in_pred[sem_seg_g] += [tmp]
                grouppred_cnt += 1

        if save_results:
            converter.labels2ply(scene_name + '.ply', 
                                 scene_point_cloud_from_blocks[:,6:], 
                                 group_pred_final.astype(np.int32))
            scene_point_cloud_from_blocks[:, 6] *= max_scene_x
            scene_point_cloud_from_blocks[:, 7] *= max_scene_y
            scene_point_cloud_from_blocks[:, 8] *= max_scene_z
            scene_point_cloud_from_blocks[:, 3:6] *= 255.0
            ins = group_pred_final.astype(np.int32)
            sem = block_pred_sem_label_list.astype(np.int32)
            sem_output = block_pred_sem_output_list
            sem_gt = scene_gt_sem_label.reshape(-1)
            ins_gt = scene_gt_ins_label.reshape(-1)
            for i in range(scene_point_cloud_from_blocks.shape[0]):
                fout_data_label.write('%f %f %f %d %d %d %f %d %d\n' % (
                    scene_point_cloud_from_blocks[i, 6],
                    scene_point_cloud_from_blocks[i, 7], 
                    scene_point_cloud_from_blocks[i, 8], 
                    scene_point_cloud_from_blocks[i, 3], 
                    scene_point_cloud_from_blocks[i, 4], 
                    scene_point_cloud_from_blocks[i, 5], 
                    sem_output[i, sem[i]], 
                    sem[i], 
                    ins[i])
                )
                fout_gt_label.write('%d %d\n' % (
                    sem_gt[i], 
                    ins_gt[i])
                )

        fout_data_label.close()
        fout_gt_label.close()
        
    fout_out_filelist.close()



    return None


if __name__ == "__main__":
    main()

