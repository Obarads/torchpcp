# torch_point_cloud
Point cloud processing with PyTorch

## Dev Environment
- OS: Ubuntu 18.04
- python: 3.7.7
- GPU: RTX2080Ti x 1
- CUDA: 10.2
- packages
  - ```
    pip install torch==1.5.0 torchvision=0.6.0
    pip install torch-cluster==1.5.6
    pip install plyfile==0.7.2
    pip install tensorboardX
    ```

## Evaluated model
- [PointNet](examples/PointNet/README.md)
  - ModelNet40

## Models of this repository
### PointNet
- [Model of this repository](examples/PointNet/README.md)
- Paper:
  - [Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. In CVPR, 2017.](https://arxiv.org/abs/1612.00593)
- Original implementation:
  - [fxia22. pointnet.pytorch. In Github repository, 2017. (url:https://github.com/fxia22/pointnet.pytorch) (access:2020/7/20)](https://github.com/fxia22/pointnet.pytorch)
  - [charlesq34. pointnet. In Github repository, 2016. (url:https://github.com/charlesq34/pointnet) (access:2020/7/20)](https://github.com/charlesq34/pointnet)
### PointNet++ (PointNet2)
- [Model of this repository](examples/PointNet2/README.md)
- Paper: 
  - [Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In NIPS, pages 5105–5114, 2017.](https://arxiv.org/abs/1706.02413)
- Original implementation:
  - [yanx27. Pointnet_Pointnet2_pytorch. In Github repository, 2019. (url:https://github.com/yanx27/Pointnet_Pointnet2_pytorch) (access:2020/7/14)](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
  - [charlesq34. pointnet2. In Github repository, 2017. (url:https://github.com/charlesq34/pointnet2) (access:2020/7/14)](https://github.com/charlesq34/pointnet2)
### ASIS (PointNet2ASIS)
- [Model of this repository](examples/PointNet2ASIS/README.md)
- Paper:
  - [Xinlong Wang, Shu Liu, Xiaoyong Shen, Chunhua Shen and Jiaya Jia. Associatively Segmenting Instances and Semantics in Point Clouds. In CVPR, 2019.](https://arxiv.org/abs/1902.09852)
- Original implementation:
  - [WXinlong. ASIS. In Github repository, 2019. (url:https://github.com/WXinlong/ASIS) (access:2020/7/16)](https://github.com/WXinlong/ASIS)

