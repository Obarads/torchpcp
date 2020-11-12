# S3DIS (Instance and semantic segmentation) with PointNet2 with ASIS
## Note
- Some python executions use [hydra](https://github.com/facebookresearch/hydra) to manage directories.
  - I call the hydra-generated directory the hydra dir.

## How to use
### Preprocessing
- Preprocess S3DIS dataset for ASIS model.
  ```bash
  python preprocess.py -d /path/to/Stanford3dDataset_v1.2_Aligned_Version -o /path/to/preprocessed_data
  ```
  - args
    - `-d`: path to Stanford3dDataset_v1.2_Aligned_Version (S3DIS dir)
    - `-o`: dir to save preprocessed S3DIS data
  -  outputs
     - `blocks` and `scenes` dir in `preprocessed` dir.

- Get object sizes per semantic label (for test).
  ```baah
  python estimate_mean_ins_size.py dataset.root=/path/to/preprocessed_data/blocks/
  ```
  - args
    - `dataset.root`: path to preprocessed data (`blocks` dir)
  - output (hydra dir)
    - `mean_ins_size.txt`

### Training
- Execute training with S3DIS dataset.
  ```bash
  python train.py dataset.root=/path/to/preprocessed_data/blocks/
  ```
  - args
    - `dataset.root`: path to preprocessed data (`blocks` dir)
  - outputs (hydra dir)
    - tensorboard event file
    - trained models (`model.path.tar` and `f_model.path.tar`)

### Test
- Evaluate trained ASIS model using S3DIS test data.
  ```bash
  python test.py dataset.root=/home/coder/databox/datasets/S3DIS/ASIS/S3DIS/4096/scenes/ general.sizetxt=/path/to/mean_ins_size.txt model.resume=/path/to/f_model.path.tar
  ```
  - args
    - `dataset.root`: path to preprocessed data (`scenes` dir)
    - `general.sizetxt`: path to`mean_ins_size.txt`
    - `model.resume`: path to `f_model.path.tar`
  - outputs (hydra dir)
    - `output_filelist.txt`
    - scene ply files
    - prediction label txt files

### Results
- Output test results.
  ```
  python eval_iou_accuracy.py -f /path/to/output_filelist.txt -r results.md
  ```
  - args
    - `-f`: path to `output_filelist.txt`
    - `-r`: path to a test result file
  - output
    - a result markdown file
