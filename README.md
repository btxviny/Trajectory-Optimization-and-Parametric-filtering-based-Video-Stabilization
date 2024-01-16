## Trajectory Optimization and Parametric filtering based Video Stabilization
This repository consists of various video stabilization methods using different motion models.1)Homography transformations between consecutive frames. 2) Sparse vertex grid motion. 3) Dense optical flow fields.

![drawn](https://github.com/btxviny/Trajectory-Optimization-and-Parametric-filtering-based-Video-Stabilization/blob/main/images/drawn_small.gif).

## Feature based stabilization.
The camera path is constructed by matching features between consecutive frames and the unwanted motion is removed with low-pass filtering.
```bash
python  scripts/stabilize_features.py --in_path unstable_video_path --out_path result_path
```
- Replace `unstable_video_path` with the path to your input unstable video.
- Replace `result_path` with the desired path for the stabilized output video.
       
## Sparse vertex grid motion optimization.
 I provide my implementation of[MeshFlow](http://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Minimum_Latency_Deep_Online_Video_Stabilization_ICCV_2023_paper.pdf).
 I provide my own gradient-based optimization method using PyTorch with a simple smoothness loss function.
 For each algorithm I provide two different warping methods. A mosaic warping method as well as a method based on [PCA-Flow](http://openaccess.thecvf.com/content_cvpr_2015/papers/Wulff_Efficient_Sparse-to-Dense_Optical_2015_CVPR_paper.pdf) which does not introduce any black borders.
```bash
python  scripts/stabilize_meshflow.py --in_path unstable_video_path --out_path result_path
```
```bash
python  scripts/stabilize_meshflow_pca.py --in_path unstable_video_path --out_path result_path
```
```bash
python  scripts/stabilize_sparse_mosaic.py --in_path unstable_video_path --out_path result_path
```
```bash
python  scripts/stabilize_sparse_pca.py --in_path unstable_video_path --out_path result_path
```
- Replace `unstable_video_path` with the path to your input unstable video.
- Replace `result_path` with the desired path for the stabilized output video.
## Dense Optical Field/ Pixel profile smoothing.
I provide a pixel profile gradient based optimization method and a parametric filtering approach.
```bash
python  scripts/stabilize_pixel_profiles_optim.py --in_path unstable_video_path --out_path result_path
```
```bash
python  scripts/stabilize_pixel_profiles_filtering.py --in_path unstable_video_path --out_path result_path
```
- Replace `unstable_video_path` with the path to your input unstable video.
- Replace `result_path` with the desired path for the stabilized output video.
     
![plot](https://github.com/btxviny/Trajectory-Optimization-and-Parametric-filtering-based-Video-Stabilization/blob/main/images/plot.png).

## Evaluation
For method evaluation and comparison I will use non-reference metrics commonly used among researchers. These metrics include: 1) cropping ratio 2) global distortion 3) pixel loss and 4) stability. We will interpret them as scores, and a good result should have a value close to 1. These metrics except the pixel loss were defined in [Bundled Camera Paths for Video Stabilization](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/Stabilization_SIGGRAPH13.pdf). For my evaluation dataset we will use the one provided in [Bundled Camera Paths for Video Stabilization](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/Stabilization_SIGGRAPH13.pdf). We take 5 videos from different categories of camera motion and scene layouts. These categories include parallax, quick rotation, zooming, crowd and regular unstable videos. The average score in each category will be used to compare methods.
The results are:
![scores](https://github.com/btxviny/Trajectory-Optimization-and-Parametric-filtering-based-Video-Stabilization/blob/main/images/scores.png)

The evaluation dataset can be downloaded [here](https://drive.google.com/file/d/1HJ0E4GIhhxtv70CIg1F9A_dZMUwDFZRq/view?usp=sharing) and the results were generated using the evaluation_all.ipynb notebook.
