# ModelNet40 (Classification) with Point Transformer
## Note
- Some python executions use [hydra](https://github.com/facebookresearch/hydra) to manage directories.
  - I call the hydra-generated directory the hydra dir.

## How to use
### Download
- Download ModelNet40 dataset ([modelnet40_normal_resampled](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)).

### (Option) Check configs
- If you want to check configs, Please execute following command.
  ```bash
  python libs/configs.py
  ```

### Preprocessing
- You must preprocess ModelNet40 dataset for time-saving training. Please check [mymodelnet40](../../tools/mymodelnet40/README.md).

### Training
- Execute training with `mymodelnet40`.
  ```bash
  python train.py dataset.root=/path/to/mymodelent40/
  ```
  - args
    - `dataset.root`: path to `mymodelnet40` dir
  - outputs (hydra dir)
    - tensorboard event file
    - trained models (`model.path.tar` and `f_model.path.tar`)

### Test Result
- Evaluate trained PointNet model using ModelNet40 test data.
  ```bash
  python test.py dataset.root=/path/to/mymodelnet40/ \
  model.resume=/path/to/f_model.path.tar
  ```
  - args
    - `dataset.root`: path to `mymodelnet40` dir
    - `model.resume`: path to `f_model.path.tar` (a trained model)



