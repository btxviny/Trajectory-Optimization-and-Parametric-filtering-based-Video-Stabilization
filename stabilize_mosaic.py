import cv2
import numpy as np
import math
import torch
import argparse
from scipy.signal import medfilt as median_filter
from time import sleep
H,W = 360,640
M,N = 5,5

device = 'cuda'
mesh_outlier_subframe_row_count = 4
mesh_outlier_subframe_col_count = 4
spatial_radius = 100
homography_min_number_corresponding_features = 4


def parse_args():
    parser = argparse.ArgumentParser(description='Video Stabilization using DMBVS-UNet')
    parser.add_argument('--in_path', type=str, help='Input video file path')
    parser.add_argument('--out_path', type=str, help='Output stabilized video file path')
    return parser.parse_args()

def fixBorder(frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1 )
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame

def draw_grid(img):
    y = np.linspace(0, H - 1, M, dtype=int)
    x = np.linspace(0, W - 1, N, dtype=int)
    grid_y, grid_x = np.meshgrid(y, x, indexing='ij')
    top_left = np.stack([grid_x[:-1, :-1], grid_y[:-1, :-1]], axis=-1)
    top_right = np.stack([grid_x[:-1, 1:], grid_y[:-1, 1:]], axis=-1)
    bottom_left = np.stack([grid_x[1:, :-1], grid_y[1:, :-1]], axis=-1)
    bottom_right = np.stack([grid_x[1:, 1:], grid_y[1:, 1:]], axis=-1)
    grid_vertices = np.stack([grid_x, grid_y],-1).reshape(-1,2)
    four_corners = np.stack([top_left, top_right, bottom_left, bottom_right], axis=-2)
    painted = img.copy()
    for i in range(four_corners.shape[0]):
        for j in range(four_corners.shape[1]):
            # Get the vertices of the current grid
            vertices = four_corners[i, j]

            # Draw the rectangle for the current grid
            cv2.rectangle(painted, tuple(vertices[0]), tuple(vertices[3]), (0, 255, 0), 1)
    return painted

def warp_frame(img, grid_motion):
    ''' Inputs: 
            Image: image to be warped of shape [H,W,3]
            grid_motion: motion of grid vertices of shape [M,N,2]
        Outputs:
            Warped_frame: warped frame
    '''
    H,W,_ = img.shape
    y = np.linspace(0, H - 1, M, dtype=int)
    x = np.linspace(0, W - 1, N, dtype=int)
    grid_y, grid_x = np.meshgrid(y, x, indexing='ij')
    grid_vertices = np.stack([grid_x, grid_y],-1)
    yy,xx = np.meshgrid(np.arange(0,H), np.arange(0,W), indexing = 'ij')
    original_grid = np.stack([xx,yy],axis = -1).astype(np.float32)
    frame_warp_field = np.zeros_like(original_grid)
    for i in range(M-1):
        for j in range(N-1):
            src = np.array([grid_vertices[i,j],grid_vertices[i,j+1],\
                            grid_vertices[i+1, j+1], grid_vertices[i+1,j]]).astype(np.float32)
            dst = src + np.array([grid_motion[i,j], grid_motion[i,j+1],\
                            grid_motion[i+1, j+1], grid_motion[i+1,j]]).astype(np.float32)
            homography, _ = cv2.findHomography(src, dst)
            x1,y1 = src[0].astype(np.int32)
            x2,y2 = src[-2].astype(np.int32)
            original_coordinates = original_grid[y1:y2, x1:x2,:]
            warped_coordinates = cv2.perspectiveTransform(original_coordinates,homography)
            frame_warp_field[y1:y2, x1:x2,:] = warped_coordinates
    warped_frame = cv2.remap(
        img,
        frame_warp_field[...,0].reshape(H,W,1).astype(np.float32),
        frame_warp_field[...,1].reshape(H,W,1).astype(np.float32),
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT
    )
    return warped_frame, frame_warp_field

def motion_estimation(frames):
    frame_count,_,_,_ = frames.shape
    y = np.linspace(0, H - 1, M, dtype=int)
    x = np.linspace(0, W - 1, N, dtype=int)
    grid_y, grid_x = np.meshgrid(y, x, indexing='ij')
    grid_vertices = np.stack([grid_x, grid_y],-1).reshape(-1,2)
    feature_detector = cv2.FastFeatureDetector_create()
    # Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    grid_motion = np.zeros((frame_count - 1, grid_vertices.shape[0],2),dtype = np.float32)
    lambda_coeff = np.ones((frame_count -1), dtype = np.float32)
    for frame_idx in range(frame_count - 1):
        prev = frames[frame_idx,...]
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr = frames[frame_idx + 1, ...]
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        #detect SIFT keypoints by sub_frames with ransac outlier removal
        frame_height, frame_width = H, W
        subframe_width = math.ceil(frame_width / mesh_outlier_subframe_col_count)
        subframe_height = math.ceil(frame_height / mesh_outlier_subframe_row_count)

        prev_features_by_subframe = []
        curr_features_by_subframe = []
        for subframe_left_x in range(0, frame_width, subframe_width):
                for subframe_top_y in range(0, frame_height, subframe_height):
                    prev_subframe = prev_gray[subframe_top_y:subframe_top_y+subframe_height,
                                            subframe_left_x:subframe_left_x+subframe_width]
                    curr_subframe = curr_gray[subframe_top_y:subframe_top_y+subframe_height,
                                            subframe_left_x:subframe_left_x+subframe_width]
                    subframe_offset = [subframe_left_x, subframe_top_y]
                    #detect keypoints in previous subframe
                    kpts = feature_detector .detect(prev_subframe)
                    kpts = np.array([kpt.pt for kpt in kpts], dtype=np.float32)
                    if len(kpts) > 0:
                        correspondances, status, _ = cv2.calcOpticalFlowPyrLK(prev_subframe, curr_subframe, kpts, None, **lk_params)
                        kpts_good = kpts[status[:,0] == 1]
                        correspondances_good = correspondances[status[:,0] == 1]
                        #filter outliers with local homography ransac
                        if len(kpts_good) > homography_min_number_corresponding_features:
                            _, outliers_mask = cv2.findHomography(kpts_good, correspondances_good,
                                                    cv2.RANSAC, ransacReprojThreshold=5.0)
                            outliers_mask = outliers_mask.flatten().astype(dtype = bool)
                            kpts_good = kpts_good[outliers_mask]
                            correspondances_good = correspondances_good[outliers_mask]
                        kpts_good += subframe_offset
                        correspondances_good += subframe_offset
                        
                        prev_features_by_subframe.append(kpts_good)
                        curr_features_by_subframe.append(correspondances_good)
        prev_features = np.concatenate(prev_features_by_subframe)
        curr_features = np.concatenate(curr_features_by_subframe)
        drawn = draw_grid(prev)
        for (x,y) in prev_features:
            cv2.circle(drawn,(int(x),int(y)), radius = 2 ,color = [0,0,255])  
        if len(prev_features_by_subframe) < homography_min_number_corresponding_features:
            continue
        #initialize motion with global homography
        mat, _ = cv2.findHomography(prev_features, curr_features)
        dx = mat[0, 2].astype(np.float32)
        dy = mat[1, 2].astype(np.float32)
        warped_vertices = np.matmul(mat , np.concatenate([grid_vertices, np.ones((grid_vertices.shape[0], 1))], axis = 1).T)
        warped_vertices = (warped_vertices[:2,...] / warped_vertices[2,...]).T
        grid_motion[frame_idx,...] = warped_vertices - grid_vertices
        #grid_motion[frame_idx,:,0] += dx / W
        #grid_motion[frame_idx,:,1] += dy / H
        #predict lambda
        translational_element  = np.sqrt((dx / W)**2 + (dy / H)**2)
        l1 = -1.93 * translational_element + 0.95
        affine_part = mat.copy()
        affine_part[2,:] = [0, 0, 1]
        eigenvalues = np.abs(np.linalg.eigvals(affine_part))
        eigenvalues = sorted(eigenvalues,reverse= True)
        affine_component = eigenvalues[0] / eigenvalues[1]
        l2 = 5.83 * affine_component - 4.88
        lambda_coeff[frame_idx] = max(min(l1,l2),0)
        # calculate vertex flows
        kpt_flow = curr_features - prev_features
        img_center= (cx,cy) = (W//2, H//2)
        vector_from_center_to_kpt = np.array([[xx - cx, yy - cy] for (xx,yy) in prev_features])
        for vertex_idx, vertex in enumerate(grid_vertices):
            (px,py) = vertex
            vector_from_vertex_to_center = np.array([cx - px, cy - py ])
            vector_from_vertex_to_kpt = vector_from_vertex_to_center + vector_from_center_to_kpt
            distances = np.linalg.norm(vector_from_vertex_to_kpt, axis = 1, keepdims = False)
            indices = np.where(distances <= spatial_radius)
            if len(indices) > 10: #only propagate motion if there are at least 10 nearby kpts
                nearby_kpt_flows = kpt_flow[indices,:]
                #apply f1 median filter to nearby flows
                kernel_size = 5
                filtered_dx = median_filter(nearby_kpt_flows[:,0], kernel_size = kernel_size)
                filtered_dy = median_filter(nearby_kpt_flows[:,1], kernel_size= kernel_size)
                filtered_flows = np.stack([filtered_dx, filtered_dy], axis = 1)
                mean_flow = np.mean(filtered_flows, axis = 0)
                grid_motion[frame_idx, vertex_idx,: ] += mean_flow
        #apply f2 filter to overall grid flow
        filtered_dx = median_filter(grid_motion[frame_idx,:,0], kernel_size= 5)
        filtered_dy = median_filter(grid_motion[frame_idx,:,1], kernel_size= 5)
        grid_motion[frame_idx,...] = np.stack([filtered_dx, filtered_dy], axis = 1)
        for (x1,y1), (dx,dy) in zip(grid_vertices[1:,...], grid_motion[frame_idx,1:,:]):
            angle = np.arctan2(dy, dx)
            magnitude = np.sqrt(dx **2 + dy ** 2)
            magnitude = np.floor(magnitude * 10).astype(np.int32)
            x2 = np.floor(x1 + magnitude * np.cos(angle)).astype(np.int32)
            y2 = np.floor(y1 + magnitude * np.sin(angle)).astype(np.int32)
            start_point = (x1,y1)
            end_point = (x2,y2)
            drawn = cv2.arrowedLine(drawn, start_point, end_point, 
                                            (255, 255, 255), 2)
        cv2.imshow('window',drawn)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break  
    cv2.destroyAllWindows()
    grid_paths = np.cumsum(grid_motion, axis = 0)
    return grid_paths, lambda_coeff

def optimize_path(grid_paths):
    original_trajectories = grid_paths
    original_trajectories = torch.from_numpy(original_trajectories).float().to(device)
    smooth_trajectories = original_trajectories.clone().requires_grad_(True)
    lr = 0.01
    optimizer = torch.optim.Adam([smooth_trajectories], lr=lr)
    ITERATIONS = 10000
    for step in range(ITERATIONS):
        # Calculate the deviation from the original trajectories
        deviation_term = 1 * torch.mean((smooth_trajectories - original_trajectories)**2)
        
        # Calculate the smoothness term
        smoothness_term =  10 * torch.mean((smooth_trajectories[1:] - smooth_trajectories[:-1])**2)
        # Balance the two terms using a parameter alpha
        loss = smoothness_term #+ deviation_term

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'\rStep {step}, Loss: {loss.item()}',end='')
    warps = smooth_trajectories - original_trajectories
    return warps.cpu().detach().numpy()

if __name__ == '__main__':
    args = parse_args()
    cap = cv2.VideoCapture(args.in_path)
    frames = []
    while True:
        ret, img = cap.read()
        if not ret: break
        img = cv2.resize(img, (W,H))
        frames.append(img)
    frames = np.array(frames,dtype = np.uint8)
    frame_count,_,_,_ = frames.shape
    grid_paths, _ = motion_estimation(frames)
    
    warps = optimize_path(grid_paths)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_path, fourcc, 30.0, (W,H))
    warps = warps.reshape(frame_count -1,M,N,2)
    for idx in range(frame_count -1):
        img = frames[idx,...]
        warp = warps[idx,...]
        warped_frame, frame_warp_field = warp_frame(img,warp) 
        warped_frame = fixBorder(warped_frame)
        out.write(warped_frame)
        cv2.imshow('window',warped_frame)
        sleep(1/30)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    out.release()
cv2.destroyAllWindows()