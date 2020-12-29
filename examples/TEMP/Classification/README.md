# ModelNet40 (Classification)
## Note
- Some python executions use [hydra](https://github.com/facebookresearch/hydra) to manage directories.
  - I call the hydra-generated directory the hydra dir.

## How to use
### Download
- Download ModelNet40 dataset ([modelnet40_ply_hdf5_2048](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)).
- Please refer to [README.md of modelnet40_ply_hdf5_2048](../../tools/modelnet40_ply_hdf5_2048/README.md).

### (Option) Check configs
- If you want to check configs, Please execute following command.
  ```bash
  python libs/configs.py
  ```

### Training
- Execute training with ModelNet40 dataset.
  ```bash
  python train.py dataset.root=/path/to/modelnet40_ply_hdf5_2048
  ```
  - args
    - `dataset.root`: path to `modelnet40_ply_hdf5_2048` dir
  - outputs (hydra dir)
    - tensorboard event file
    - trained models (`model.path.tar` and `f_model.path.tar`)

### Test Result
- Evaluate trained PointNet model using ModelNet40 test data.
  ```bash
  python test.py dataset.root=/path/to/modelnet40_ply_hdf5_2048/ \
  model.resume=/path/to/f_model.path.tar
  ```
  - args
    - `dataset.root`: path to `modelnet40_ply_hdf5_2048` dir
    - `model.resume`: path to `f_model.path.tar` (a trained model)

