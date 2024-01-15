## Trajectory Optimization and Parametric filtering based Video Stabilization
This repository consists of various video stabilization methods using different motion models.1)Homography transformations between consecutive frames. 2) Sparse vertex grid motion. 3) Dense optical flow fields.

![drawn](https://github.com/btxviny/Trajectory-Optimization-and-Parametric-filtering-based-Video-Stabilization/blob/main/images/drawn_small.gif).

## Feature based stabilization.
The camera path is constructed by matching features between consecutive frames and the unwanted motion is removed with low-pass filtering.
```bash
python stabilize_features.py --in_path unstable_video_path --out_path result_path
```
       
## Sparse vertex grid motion optimization.
 I provide my implementation of[MeshFlow](http://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Minimum_Latency_Deep_Online_Video_Stabilization_ICCV_2023_paper.pdf).
 I provide my own gradient-based optimization method using PyTorch with a simple smoothness loss function.
 For each algorithm I provide two different warping methods. A mosaic warping method as well as a method based on [PCA-Flow](http://openaccess.thecvf.com/content_cvpr_2015/papers/Wulff_Efficient_Sparse-to-Dense_Optical_2015_CVPR_paper.pdf) which does not introduce any black borders.
```bash
python stabilize_meshflow.py --in_path unstable_video_path --out_path result_path
```
```bash
python stabilize_meshflow_pca.py --in_path unstable_video_path --out_path result_path
```
```bash
python stabilize_sparse_mosaic.py --in_path unstable_video_path --out_path result_path
```
```bash
python stabilize_sparse_pca.py --in_path unstable_video_path --out_path result_path
```
- Replace `unstable_video_path` with the path to your input unstable video.
- Replace `result_path` with the desired path for the stabilized output video.
## Dense Optical Field/ Pixel profile smoothing.
I provide a pixel profile gradient based optimization method and a parametric filtering approach.
```bash
python stabilize_pixel_profiles_optim.py --in_path unstable_video_path --out_path result_path
```
```bash
python stabilize_pixel_profiles_filtering.py --in_path unstable_video_path --out_path result_path
```
     
![plot](https://github.com/btxviny/Trajectory-Optimization-and-Parametric-filtering-based-Video-Stabilization/blob/main/images/plot.png).
