# Torchpcp

[![python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT%202.0-g.svg)](LICENSE)

## About
PyTorch point cloud is a pytorch implementation collection of point cloud utils, models, and modules.

## Component
The Componets of this repository is as following.
- `examples`: include scripts to train or test implemented models.
  - Test results of models are shown on README.md (ex: [examples/PointNet/README.md](examples/PointNet/README.md)) in the model folder.
    - Please look at a list in ["Evaluated model" section](#evaluated-model) when checking evaluated models.
  - Example codes work by adding `TORCHPCP_DEBUG=true` to environment variable, even if you don't install torchpcp package using [setup.py](setup.py).
- `torchpcp`: include model, module, and utils.
  - Docstring guide of this repo follows [numpy](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).

## Dependencies
- Python3 (3.7.7)
- PyTorch (1.7.0)
- Ninja (1.8.2)
  - If you have not installed this software, Please show [here](https://www.claudiokuenzler.com/blog/756/install-newer-ninja-build-tools-ubuntu-14.04-trusty).
- (optional) [Hydra](https://github.com/facebookresearch/hydra) (1.0 or later)
- (optional) [tensorboardX](https://github.com/lanpa/tensorboardX) (2.1 or later)

## Install
```
python setup.py install
```

## Execution environment
- Docker: nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
- GPU: RTX2080Ti x 1

## Evaluated model
- The following models were trained and tested in my environment.
- README.md of the following models shows the results of the model in implementations of this repository and papers.
- [PointNet](examples/PointNet/README.md)
  - [Classification (ModelNet40)](examples/PointNet/Classification/README.md)
- [PointNet2ASIS (ASIS)](examples/PointNet2ASIS/README.md)
  - [Instance and semantic segmentation (S3DIS Area5)](examples/PointNet2ASIS/Ins.Sem.Seg./README.md)
- [DGCNN](examples/DGCNN/README.md)
  - [Classification (ModelNet40)](examples/DGCNN/Classification/README.md)

## Models of this repository
### PointNet
- [Model of this repository](examples/PointNet/README.md)
- Paper:
  - [Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. In CVPR, 2017.](https://arxiv.org/abs/1612.00593)
- Original implementation:
  - [fxia22. pointnet.pytorch. In Github repository. (url:https://github.com/fxia22/pointnet.pytorch) (access:2020/7/20)](https://github.com/fxia22/pointnet.pytorch)
  - [charlesq34. pointnet. In Github repository. (url:https://github.com/charlesq34/pointnet) (access:2020/7/20)](https://github.com/charlesq34/pointnet)
### PointNet++ (PointNet2)
- [Model of this repository](examples/PointNet2/README.md)
- Paper: 
  - [Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In NIPS, pages 5105–5114, 2017.](https://arxiv.org/abs/1706.02413)
- Original implementation:
  - [yanx27. Pointnet_Pointnet2_pytorch. In Github repository. (url:https://github.com/yanx27/Pointnet_Pointnet2_pytorch) (access:2020/7/14)](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
  - [charlesq34. pointnet2. In Github repository. (url:https://github.com/charlesq34/pointnet2) (access:2020/7/14)](https://github.com/charlesq34/pointnet2)
### ASIS (PointNet2ASIS)
- [Model of this repository](examples/PointNet2ASIS/README.md)
- Paper:
  - [Xinlong Wang, Shu Liu, Xiaoyong Shen, Chunhua Shen and Jiaya Jia. Associatively Segmenting Instances and Semantics in Point Clouds. In CVPR, 2019.](https://arxiv.org/abs/1902.09852)
- Original implementation:
  - [WXinlong. ASIS. In Github repository. (url:https://github.com/WXinlong/ASIS) (access:2020/7/16)](https://github.com/WXinlong/ASIS)
### DGCNN
- [Model of this repository](examples/DGCNN/README.md)
- Paper:
  - [Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. Dynamic Graph CNN for Learning on Point Clouds. In ACM Transactions on Graphics (TOG), 2019.](https://arxiv.org/abs/1801.07829)
- Original implementation:
  - [WangYueFt. dgcnn. In Github repository. (url:https://github.com/WangYueFt/dgcnn) (access:2020/11/14)](https://github.com/WangYueFt/dgcnn)
### PointCNN
- [Model of this repository](examples/PointCNN/README.md)
- Paper:
  - [Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. PointCNN: Convolution On X-Transformed Points. In NIPS, 2018.](https://arxiv.org/abs/1801.07829)
- Original implementation:
  - [yangyanli. PointCNN. In Github repository. (https://github.com/yangyanli/PointCNN) (access:2020/11/21)](https://github.com/yangyanli/PointCNN)
### KPCNN
- [Model of this repository](examples/KPCNN/README.md)
- Paper:
  - [Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette, and Leonidas J. Guibas. KPConv: Flexible and Deformable Convolution for Point Clouds. In ICCV, 2019.](https://arxiv.org/abs/1904.08889)
- Original implementation:
  - [HuguesTHOMAS. KPConv-PyTorch. In Github repository. (https://github.com/HuguesTHOMAS/KPConv-PyTorch) (access:2020/11/21)](https://github.com/HuguesTHOMAS/KPConv-PyTorch)
### Point Transformer
- [Model of this repository](examples/PointTransformer/README.md)
- Paper:
  - [Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip Torr, Vladlen Koltun. Point Transformer. arxiv.](https://arxiv.org/abs/2012.09164)


