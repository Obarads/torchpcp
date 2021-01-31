import torch
from torch import nn

from torchpcp.modules.Layer import Conv1DModule
# from torchpcp.modules.Sampling import knn_index, index2points
from torchpcp.modules.functional.nns import k_nearest_neighbors
from torchpcp.modules.functional.other import index2points

class ASIS(nn.Module):
    def __init__(self, sem_in_channels, sem_out_channels, ins_in_channels, 
                 ins_out_channels, k, memory_saving=True):
        super(ASIS, self).__init__()

        # sem branch
        self.sem_pred_fc = nn.Sequential(
            nn.Dropout(inplace=True),
            nn.Conv1d(sem_in_channels, sem_out_channels, 1)
        ) # input: F_ISEM, output: P_SEM

        # interactive module: sem to ins
        self.adaptation = Conv1DModule(sem_in_channels, ins_in_channels)

        # ins branch
        self.ins_emb_fc = nn.Sequential(
            nn.Dropout(inplace=True),
            nn.Conv1d(ins_in_channels, ins_out_channels, 1)
        ) # input: F_SINS, output: E_INS

        # interactive module: ins to sem
        # using knn_index and index2points

        self.k = k
        self.memory_saving = memory_saving

    def forward(self, f_sem, f_ins):
        adapted_f_sem = self.adaptation(f_sem)

        # for E_INS
        f_sins = f_ins + adapted_f_sem
        e_ins = self.ins_emb_fc(f_sins)

        # for P_SEM
        nn_idx, _ = k_nearest_neighbors(e_ins, e_ins, self.k)
        k_f_sem = index2points(f_sem, nn_idx)
        f_isem = torch.max(k_f_sem, dim=3, keepdim=True)[0]
        f_isem = torch.squeeze(f_isem, dim=3)
        p_sem = self.sem_pred_fc(f_isem)

        return p_sem, e_ins