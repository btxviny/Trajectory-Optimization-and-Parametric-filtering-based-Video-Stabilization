## Trajectory Optimization and Parametric filtering based Video Stabilization
This repository consists of various video stabilization methods using different motion models.1)Homography transformations between consecutive frames. 2) Sparse vertex grid motion. 3) Dense optical flow fields.
In the sparse vertex grid approaches I provide a mosaic warping method as well as a method based on [PCA-Flow](http://openaccess.thecvf.com/content_cvpr_2015/papers/Wulff_Efficient_Sparse-to-Dense_Optical_2015_CVPR_paper.pdf).

![drawn](https://github.com/btxviny/Trajectory-Optimization-Video-Stabilization/blob/main/images/drawn_small.gif).

## Feature based stabilization.
The camera path is constructed by matching features between consecutive frames and the unwanted motion is removed with low-pass filtering.
-feature_stabilization.py
       
## Sparse vertex grid motion optimization.
 I provide my implementation of[MeshFlow](http://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Minimum_Latency_Deep_Online_Video_Stabilization_ICCV_2023_paper.pdf).
 I provide my own gradient-based optimization method using PyTorch with a simple smoothness loss function.
 For each algorithm I provide two different warping methods. A mosaic warping method as well as a method based on [PCA-Flow](http://openaccess.thecvf.com/content_cvpr_2015/papers/Wulff_Efficient_Sparse-to-
 
## Dense_Optical_2015_CVPR_paper.pdf) which does not introduce any black borders.
- meshflow_mosaic.ipynb
- meshflow_pca.ipynb
- sparse_grid.ipynb
- sparse_grid_pca.ipynb
## Dense Optical Field/ Pixel profile smoothing.
I provide a pixel profile gradient based optimization method and a parametric filtering approach.
-dense_optim.ipynb
-dense_parametric.ipynb
     
![plot](https://github.com/btxviny/Trajectory-Optimization-Video-Stabilization/blob/main/images/plot.png).
