## Trajectory Optimization and Parametric filtering based Video Stabilization
This repository consists of various video stabilization methods using different motion models.1)Homography transformations between consecutive frames. 2) Sparse vertex grid motion. 3) Dense optical flow fields.
In the sparse vertex grid approaches I provide a mosaic warping method as well as a method based on (PCA-Flow)[http://openaccess.thecvf.com/content_cvpr_2015/papers/Wulff_Efficient_Sparse-to-Dense_Optical_2015_CVPR_paper.pdf]

##Feature based stabilization.
     - The camera path is constructed by matching features between consecutive frames and the unwanted motion is removed with low-pass filtering. This is implemented in feature_stabilization.py
       
## Sparse vertex grid motion optimization as described in [MeshFlow](http://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Minimum_Latency_Deep_Online_Video_Stabilization_ICCV_2023_paper.pdf).
- ![mosaic]
Sparse vertex grid motion optimization using a gradient-based  method using PyTorch with a simple smoothness loss function.
Dense optical flow / pixel profile optimization.
Dense optical flow parametric filtering.

![drawn](https://github.com/btxviny/Trajectory-Optimization-Video-Stabilization/blob/main/images/drawn_small.gif).

The resulting optimized path is:

![plot](https://github.com/btxviny/Trajectory-Optimization-Video-Stabilization/blob/main/images/plot.png).

And the results are:
![result](https://github.com/btxviny/Trajectory-Optimization-Video-Stabilization/blob/main/images/concatenated.gif).
##Instructions
- Run the following command:
     ```bash
     python stabilize_meshflow.py --in_path input.avi --out_path output.avi
     ```
     or
    ```bash
    python stabilize_gradient_optimization.py --in_path input.avi --out_path output.avi
    ```
   -Replace ./input.avi with the path to your unstable video.
  
   -Replace ./output.avi with the path to where you want to store the result.
