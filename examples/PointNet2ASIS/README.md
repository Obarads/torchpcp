# PointNet++ with ASIS
## How to use
- I separate training and test codes into each task folder. Please find below, folder links of each task.

## Links
### Instance and semantic segmentation
- [code](./Ins.Sem.Seg./README.md)
- **S3DIS Test on Area5**
  - Semantic Segmentation   
    |Impl.|mAcc|mIoU|oAcc|
    |-----|----|----|----|
    |[Original](https://github.com/WXinlong/ASIS)(vanilla)|58.3|50.8|86.7|
    |[Original](https://github.com/WXinlong/ASIS)|60.9|53.4|86.9|
    |Model of this repo.|62.4|54.5|87.2|
  - Instance Segmentation  
    |Impl.|mCov|mWCov|mPrec|mRec|
    |-----|----|-----|-----|----|
    |[Original](https://github.com/WXinlong/ASIS)(vanilla)|42.6|45.7|53.4|40.6|
    |[Original](https://github.com/WXinlong/ASIS)|44.6|47.8|55.3|42.4|
    |Model of this repo.|47.5|50.5|56.9|44.3|
- **S3DIS Test on 6-fold CV**
  - Semantic Segmentation   
    |Impl.|mAcc|mIoU|oAcc|
    |-----|----|----|----|
    |[Original](https://github.com/WXinlong/ASIS)(vanilla)|69.0|58.2|85.9|
    |[Original](https://github.com/WXinlong/ASIS)|70.1|59.3|86.2|
    |Model of this repo.|yet|yet|yet|
  - Instance Segmentation  
    |Impl.|mCov|mWCov|mPrec|mRec|
    |-----|----|-----|-----|----|
    |[Original](https://github.com/WXinlong/ASIS)(vanilla)|49.6|53.4|62.7|45.8|
    |[Original](https://github.com/WXinlong/ASIS)|51.2|55.1|63.6|47.5|
    |Model of this repo.|yet|yet|yet|yet|

