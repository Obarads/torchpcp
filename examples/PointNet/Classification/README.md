# ModelNet40 (classification) with PointNet
## How to use
### Training
- ```bash
  python train.py dataset.root=$PREPRODATA_DIR dataset.name=modelnet40
  ```
  - args
    - `dataset.root`: select dataset directory path
      - `$PREPRODATA_DIR`: a environment variable of dataset directory path
## Results

|Impl.|mAcc|oAcc|
|-----|----|----|
|[Original](https://github.com/charlesq34/pointnet)|86.2|89.2|
|Our|84.4|87.9|


## References
### Documents
- [Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. In CVPR, 2017.](https://arxiv.org/abs/1612.00593)

### Implementation
- [fxia22. pointnet.pytorch. In Github repository, 2017. (url:https://github.com/fxia22/pointnet.pytorch) (access:2020/7/20)](https://github.com/fxia22/pointnet.pytorch)
- [charlesq34. pointnet. In Github repository, 2017. (url:https://github.com/charlesq34/pointnet) (access:2020/7/20)](https://github.com/charlesq34/pointnet)

### Note
- I replace AccMeter2 with AssessmentMeter(S3DISMeter(metric="class")). 2020/9/27
- I remove AccMeter2. 2020/9/27

