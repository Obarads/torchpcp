# torchpcp_cpp
## About
- These codes are PyTorch C++ implementation for point cloud processing.

## Dependencies
- PyTorch (1.7.0)
- Ninja (1.8.2)
  - If you have not installed this software, Please show [here](https://www.claudiokuenzler.com/blog/756/install-newer-ninja-build-tools-ubuntu-14.04-trusty).

## Codes
### Ball query
- link:[src/ball_query/](src/ball_query/)
- Original implementation:
  - [mit-han-lab/pvcnn. (url:https://github.com/mit-han-lab/pvcnn) (access:2020/11/7)](https://github.com/mit-han-lab/pvcnn)
- Note: because there is computing very small numbers below decimal point, ball_query() ≠ py_ball_query().

### Grouping
- link:[src/grouping](src/grouping)
- Original implementation:
  - [mit-han-lab/pvcnn. (url:https://github.com/mit-han-lab/pvcnn) (access:2020/11/7)](https://github.com/mit-han-lab/pvcnn)

### Gather
- link:[src/gather](src/gather/)

### Interpolate
- link:[src/interpolate](src/interpolate/)
- Original implementation:
  - [mit-han-lab/pvcnn. (url:https://github.com/mit-han-lab/pvcnn) (access:2020/11/7)](https://github.com/mit-han-lab/pvcnn)

### KNN
- link:[src/knn](src/knn/)
- Note: If there are multiple values with the same distance in knn, the order of the indexes may change.

### Sampling
- link:[src/interpolate](src/sampling)
- Original implementation:
  - [mit-han-lab/pvcnn. (url:https://github.com/mit-han-lab/pvcnn) (access:2020/11/7)](https://github.com/mit-han-lab/pvcnn)

### Voxelization
- link:[src/voxelization](src/voxelization/)
- Original implementation:
  - [mit-han-lab/pvcnn. (url:https://github.com/mit-han-lab/pvcnn) (access:2020/11/7)](https://github.com/mit-han-lab/pvcnn)

