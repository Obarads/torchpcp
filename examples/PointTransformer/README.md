# Point Transformer
## How to use
- I separate training and test codes into each task folder. Please find below, folder links of each task.

## Attention
- I can't confirm the performance of the paper in my implementation model yet.

## Note (Paper info.)
### Point Transformer layer
- \gamma : an MLP with two linear layers and one ReLU nonlinearity.
- \theta : an MLP with two linear layers and one ReLU nonlinearity.

### Transition down
- k : 16
- mlp : Linear transformation, bath normalization and ReLU.

## Links
### Classification
- [code](./Classification/README.md)
- **ModelNet40**  
    |Impl.|mAcc|oAcc|
    |-----|----|----|
    |[Original (paper only)](https://arxiv.org/abs/2012.09164)|90.6|93.7|
    |Model of this repo.|87.40|90.19|
