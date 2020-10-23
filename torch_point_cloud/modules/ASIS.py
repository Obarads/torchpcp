import torch
from torch import nn

from torch_point_cloud.modules.Layer import MLP1D
from torch_point_cloud.modules.Sampling import knn_index, index2points

class ASIS(nn.Module):
    def __init__(self, sem_in_channels, sem_out_channels, ins_in_channels, 
                 ins_out_channels, k):
        super(ASIS, self).__init__()

        # sem branch
        self.sem_fc = MLP1D(sem_in_channels, 128) # for F_SEM(?)
        self.sem_pred_fc = nn.Sequential(
            nn.Dropout(inplace=True),
            nn.Conv1d(128, sem_out_channels, 1)
        ) # input: F_ISEM, output: P_SEM

        # interactive module: sem to ins
        f_sins_num_channels = 128 # when element_wise_addition
        self.adaptation = MLP1D(128, f_sins_num_channels)

        # ins branch
        self.ins_fc = MLP1D(ins_in_channels, f_sins_num_channels) # for F_INS(?)
        self.ins_emb_fc = nn.Sequential(
            nn.Dropout(inplace=True),
            nn.Conv1d(f_sins_num_channels, ins_out_channels, 1)
        ) # input: F_SINS, output: E_INS

        # interactive module: ins to sem
        # using knn_index and index2points

        self.k = k

    def forward(self, f_sem, f_ins):
        f_sem = self.sem_fc(f_sem)
        adapted_f_sem = self.adaptation(f_sem)

        # for E_INS
        f_ins = self.ins_fc(f_ins)
        f_sins = f_ins + adapted_f_sem
        e_ins = self.ins_emb_fc(f_sins)

        # for P_SEM
        nn_idx = knn_index(e_ins, self.k, memory_saving=True)
        k_f_sem = index2points(f_sem, nn_idx)
        f_isem = torch.max(k_f_sem, dim=3, keepdim=True)[0]
        f_isem = torch.squeeze(f_isem, dim=3)
        p_sem = self.sem_pred_fc(f_isem)

        return p_sem, e_ins