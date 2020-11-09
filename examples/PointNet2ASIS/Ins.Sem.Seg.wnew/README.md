# ModelNet40 (classification) with PointNet2
## How to use
### Training
- ```bash
  python train.py dataset.root=$PREPRODATA_DIR dataset.name=ModelNet40
  ```
  - args
    - `dataset.root`: select dataset directory path
      - `$PREPRODATA_DIR`: a environment variable of dataset directory path
