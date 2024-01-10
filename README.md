## Trajectory Optimization and Parametric filtering based Video Stabilization
This repository consists of various video stabilization methods.
##Feature based stabilization.
- The camera path is constructed by matching features between consecutive frames and the unwanted motion is removed with low-pass filtering.
Sparse vertex grid motion optimization as described in [MeshFlow](http://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Minimum_Latency_Deep_Online_Video_Stabilization_ICCV_2023_paper.pdf).
Sparse vertex grid motion optimization using a gradient-based  method using PyTorch with a simple smoothness loss function.
Dense optical flow / pixel profile optimization.
Dense optical flow parametric filtering.

The motion is modeled in a sparse vertex grid as shown in the figure below:
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
