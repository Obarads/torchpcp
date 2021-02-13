import numpy as np

import torch
from torch.utils.data import DataLoader

# local package
from libs import tpcpath
from libs.dataset import SimpleSceneDataset
from libs.three_nn import three_nn # PointRCNN

# torch-points-kernels
import torch_points_kernels as tpk

# torchpcp pacakage
from torchpcp.modules.functional.other import index2points
from torchpcp.modules.functional.sampling import furthest_point_sampling
from torchpcp.modules.functional import nns
from torchpcp.utils.monitor import timecheck
from torchpcp.utils import pytorch_tools

# pytorch_tools.set_seed(0)
device = pytorch_tools.select_device("cuda")

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

# get dataset
dataset = SimpleSceneDataset()
points, sem_label, ins_label = dataset[0]
pc = torch.tensor([points[:, :3]], device="cuda").transpose(1,2)

# compare ball query
k = 20
d = 0.1
idx, res_dists = nns.py_ball_query(d, k, pc, pc)
idx2, res_dists2 = nns.ball_query(pc, pc, d, k)
# print(idx)
# print(idx2)
# print(dist)
check_idx = idx == idx2
if True:
    for ib in range(len(check_idx)):
        b_idxs = check_idx[ib]
        for i_n in range(len(b_idxs)):
            n_idxs = b_idxs[i_n]
            if False in n_idxs:
                for i_p in range(len(n_idxs)):
                    k_idxs = n_idxs[i_p]
                    if False == k_idxs:
                        print("pybq ib {}, in {}, ip {} dist {}".format(ib, i_n, i_p, res_dists[ib, i_n, idx[ib, i_n, i_p]]))
                        print("pybq and dist2 {}".format(res_dists[ib, i_n, idx2[ib, i_n, i_p]]))
                        print("dist {}".format(res_dists2[ib, i_n, i_p]))
                    else:
                        print("pybq ib {}, in {}, ip {} dist {}".format(ib, i_n, i_p, res_dists[ib, i_n, idx[ib, i_n, i_p]]))
# ↑ result
"""
pybq ib 0, in 201, ip 0 dist 0.0
pybq ib 0, in 201, ip 1 dist 0.006920814514160156
pybq ib 0, in 201, ip 2 dist 0.0076847076416015625
pybq ib 0, in 201, ip 3 dist 0.0037059783935546875
pybq ib 0, in 201, ip 4 dist 0.008916854858398438
pybq ib 0, in 201, ip 5 dist 0.0053310394287109375
pybq ib 0, in 201, ip 6 dist 0.0044994354248046875
pybq ib 0, in 201, ip 7 dist 0.0069255828857421875
pybq ib 0, in 201, ip 8 dist 0.006600379943847656
pybq ib 0, in 201, ip 9 dist 0.0007305145263671875
pybq ib 0, in 201, ip 10 dist 0.0022897720336914062
pybq ib 0, in 201, ip 11 dist 0.0041522979736328125
pybq ib 0, in 201, ip 12 dist 0.008325576782226562
pybq ib 0, in 201, ip 13 dist 0.0008840560913085938
dist 0.009999999776482582
pybq ib 0, in 201, ip 14 dist 0.005504608154296875
dist 0.0008840001537464559
pybq ib 0, in 201, ip 15 dist 0.005780220031738281
dist 0.005506000481545925
pybq ib 0, in 201, ip 16 dist 0.006201744079589844
dist 0.005780001170933247
pybq ib 0, in 201, ip 17 dist 0.0015707015991210938
dist 0.006202000193297863
pybq ib 0, in 201, ip 18 dist 0.006770133972167969
dist 0.0015709996223449707
pybq ib 0, in 201, ip 19 dist 0.0024690628051757812
dist 0.0067709991708397865
pybq ib 0, in 1743, ip 0 dist 0.003673553466796875
pybq ib 0, in 1743, ip 1 dist 0.006389141082763672
pybq ib 0, in 1743, ip 2 dist 0.004160404205322266
pybq ib 0, in 1743, ip 3 dist 0.007565021514892578
pybq ib 0, in 1743, ip 4 dist 0.003292083740234375
pybq ib 0, in 1743, ip 5 dist 0.004410266876220703
pybq ib 0, in 1743, ip 6 dist 0.0011606216430664062
pybq ib 0, in 1743, ip 7 dist 0.0
pybq ib 0, in 1743, ip 8 dist 0.006091594696044922
pybq ib 0, in 1743, ip 9 dist 0.0031538009643554688
pybq ib 0, in 1743, ip 10 dist 0.003829479217529297
pybq ib 0, in 1743, ip 11 dist 0.004717350006103516
dist 0.009999987669289112
pybq ib 0, in 1743, ip 12 dist 0.003985881805419922
dist 0.004717977251857519
pybq ib 0, in 1743, ip 13 dist 0.0008730888366699219
dist 0.00398601358756423
pybq ib 0, in 1743, ip 14 dist 0.007565021514892578
dist 0.0008729968103580177
pybq ib 0, in 1743, ip 15 dist 0.001956462860107422
dist 0.007564985193312168
pybq ib 0, in 1743, ip 16 dist 0.003673553466796875
dist 0.00195599184371531
pybq ib 0, in 1743, ip 17 dist 0.003673553466796875
pybq ib 0, in 1743, ip 18 dist 0.003673553466796875
pybq ib 0, in 1743, ip 19 dist 0.003673553466796875
pybq ib 0, in 2100, ip 0 dist 0.0015821456909179688
pybq ib 0, in 2100, ip 1 dist 0.005949497222900391
pybq ib 0, in 2100, ip 2 dist 0.007802486419677734
pybq ib 0, in 2100, ip 3 dist 0.0003647804260253906
pybq ib 0, in 2100, ip 4 dist 0.003986358642578125
pybq ib 0, in 2100, ip 5 dist 0.009930133819580078
pybq ib 0, in 2100, ip 6 dist 0.006264686584472656
pybq ib 0, in 2100, ip 7 dist 0.007051944732666016
dist 0.009999987669289112
pybq ib 0, in 2100, ip 8 dist 0.0
dist 0.007052002940326929
pybq ib 0, in 2100, ip 9 dist 0.0021381378173828125
dist 0.0
pybq ib 0, in 2100, ip 10 dist 0.0016779899597167969
dist 0.0021379967220127583
pybq ib 0, in 2100, ip 11 dist 0.00973367691040039
dist 0.0016780020669102669
pybq ib 0, in 2100, ip 12 dist 0.005712985992431641
dist 0.009732990525662899
pybq ib 0, in 2100, ip 13 dist 0.0016579627990722656
dist 0.005713010206818581
pybq ib 0, in 2100, ip 14 dist 0.0003647804260253906
dist 0.0016580007504671812
pybq ib 0, in 2100, ip 15 dist 0.00982522964477539
dist 0.00036500205169431865
pybq ib 0, in 2100, ip 16 dist 0.005396366119384766
dist 0.009824997745454311
pybq ib 0, in 2100, ip 17 dist 0.0015821456909179688
dist 0.00539599871262908
pybq ib 0, in 2100, ip 18 dist 0.0015821456909179688
pybq ib 0, in 2100, ip 19 dist 0.0015821456909179688
pybq ib 0, in 3347, ip 0 dist 0.0021219253540039062
dist 0.009999999776482582
pybq ib 0, in 3347, ip 1 dist 0.0090789794921875
dist 0.0021210000850260258
pybq ib 0, in 3347, ip 2 dist 0.00902557373046875
dist 0.00907999835908413
pybq ib 0, in 3347, ip 3 dist 0.007811546325683594
dist 0.009027000516653061
pybq ib 0, in 3347, ip 4 dist 0.002498626708984375
dist 0.007812999188899994
pybq ib 0, in 3347, ip 5 dist 0.009325027465820312
dist 0.002500000176951289
pybq ib 0, in 3347, ip 6 dist 0.009125709533691406
dist 0.009323998354375362
pybq ib 0, in 3347, ip 7 dist 0.005329132080078125
dist 0.00912499986588955
pybq ib 0, in 3347, ip 8 dist 0.009697914123535156
dist 0.005329999141395092
pybq ib 0, in 3347, ip 9 dist 0.006146430969238281
dist 0.009698999114334583
pybq ib 0, in 3347, ip 10 dist 0.008111953735351562
dist 0.00614599883556366
pybq ib 0, in 3347, ip 11 dist 0.0049648284912109375
dist 0.008112997747957706
pybq ib 0, in 3347, ip 12 dist 0.0
dist 0.004964999854564667
pybq ib 0, in 3347, ip 13 dist 0.0016241073608398438
dist 0.0
pybq ib 0, in 3347, ip 14 dist 0.005043983459472656
dist 0.0016249999171122909
pybq ib 0, in 3347, ip 15 dist 0.0006656646728515625
dist 0.005044000223278999
pybq ib 0, in 3347, ip 16 dist 0.0036096572875976562
dist 0.0006659999489784241
pybq ib 0, in 3347, ip 17 dist 0.009525299072265625
dist 0.003610999556258321
pybq ib 0, in 3347, ip 18 dist 0.0021219253540039062
dist 0.009524999186396599
pybq ib 0, in 3347, ip 19 dist 0.0021219253540039062
dist 0.009999999776482582
"""

# print(check_idx)
# print("res_dists:", res_dists)
# print("res_dists2:", res_dists2)
# print("ci:", check_idx)

print("CHECK2:", False in check_idx)
