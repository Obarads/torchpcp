# mymodelnet40
## About
- Prerocessed dataset of [modelnet40_normal_resampled](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) for time-saving training.
- The following models use this dataset.
  - [PointTransformer](../../PointTransformer/README.md)

## How to use
### Download
- Download ModelNet40 dataset ([modelnet40_normal_resampled](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)).

### Preprocessing
- Execute preprocessing with modelnet40_normal_resampled.
  ```bash
  python preprocessing.py -p /path/to/modelnet40_normal_resampled -o /path/to/output dir/
  ```
  - args
    - `-p`: path to `modelnet40_normal_resampled` dir
    - '-o': output path.
  - outputs
    - `mymodelnet40` dir
