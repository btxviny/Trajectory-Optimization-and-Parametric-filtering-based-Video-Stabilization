import cv2
import numpy as np
import math

def stabilize_video(frames,smoothing = 'mva'):

    def fixBorder(frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.2)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Variables for feature tracking
    prev_frame = None
    prev_keypoints = None
    prev_descriptors = None
    transforms = np.zeros((len(frames), 3))  # Tracks the cumulative motion
    n_frames, h, w, c = frames.shape

    # Iterate over frames
    for i, frame in enumerate(frames):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Detect keypoints and compute descriptors for current frame
            keypoints, descriptors = sift.detectAndCompute(gray, None)

            if prev_keypoints is not None and descriptors is not None:
                # Create a BFMatcher object
                matcher = cv2.BFMatcher()
                matches = matcher.knnMatch(prev_descriptors, descriptors, k=2)

                # Apply ratio test to filter good matches
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                if len(good_matches) > 10:
                    src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Calculate the homography matrix using RANSAC
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)

                    if M is not None:
                        dx = M[0, 2]
                        dy = M[1, 2]
                        # Extract rotation angle
                        da = np.arctan2(M[1, 0], M[0, 0])
                        # Store transformation
                        transforms[i] = [dx, dy, da]

            # Update the previous keypoints and descriptors
            prev_keypoints = keypoints
            prev_descriptors = descriptors

        # Update the previous frame
        prev_frame = gray

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = np.zeros_like(trajectory)

    if smoothing == 'mva':
        window_size = 60
        for idx in range(3):
            smoothed_trajectory[:, idx] = np.convolve(trajectory[:, idx], np.ones(window_size) / window_size,
                                                      mode='same')
    elif smoothing == 'ewma':
        # Apply exponential weighted moving average (EWMA) smoothing to the trajectory
        alpha = 0.1  # Smoothing factor
        smoothed_trajectory[0] = trajectory[0]  # Initialize the first value

        for i in range(1, len(trajectory)):
            smoothed_trajectory[i] = alpha * trajectory[i] + (1 - alpha) * smoothed_trajectory[i - 1]

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    smooth_frames = np.zeros_like(frames)
    # Write n_frames-1 transformed frames
    for i in range(n_frames - 1):
        # Read next frame
        frame = frames[i, ...]
        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        M = np.zeros((3, 3), np.float32)
        M[0, 0] = np.cos(da)
        M[0, 1] = -np.sin(da)
        M[1, 0] = np.sin(da)
        M[1, 1] = np.cos(da)
        M[0, 2] = dx
        M[1, 2] = dy
        M[2, 2] = 1  # Set the third row for homogenous coordinates

        # Apply homography transformation to the given frame
        frame_stabilized = cv2.warpPerspective(frame, M, (w, h))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)
        smooth_frames[i, ...] = frame_stabilized
    smooth_frames[-1, ...] = frames[-1, ...]
    return smooth_frames

def stabilize_video2(frames):
    def fixBorder(frame,croping):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, croping)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame
    feature_detector  = cv2.xfeatures2d.SIFT_create(100)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    frame_count, height, width, _ = frames.shape
    subframe_width = math.ceil(width / 4)
    subframe_height = math.ceil(height / 4)

    transforms = np.zeros((frame_count, 3))
    
    for frame_idx in range(frame_count - 1):
        print(f'\rEstimating trajectory {frame_idx + 1}/{frame_count}',end = '')
        prev = frames[frame_idx,...]
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr = frames[frame_idx + 1, ...]
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        prev_features_by_subframe = []
        curr_features_by_subframe = []
        for subframe_left_x in range(0, width, subframe_width):
                for subframe_top_y in range(0, height, subframe_height):
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
                        if len(kpts_good) > 4:
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
        mat, _ = cv2.findHomography(prev_features, curr_features)
        if mat is not None:
            dx = mat[0, 2]
            dy = mat[1, 2]
            # Extract rotation angle
            da = np.arctan2(mat[1, 0], mat[0, 0])
            # Store transformation
            transforms[frame_idx] = [dx, dy, da]
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = np.zeros_like(trajectory)
    window_size = 40
    for idx in range(3):
        smoothed_trajectory[:, idx] = np.convolve(trajectory[:, idx], np.ones(window_size) / window_size,
                                                    mode='same')
    window_size = 15
    for idx in range(3):
        smoothed_trajectory[:, idx] = np.convolve(smoothed_trajectory[:, idx], np.ones(window_size) / window_size,
                                                    mode='same')
    # Calculate difference in smoothed_trajectory and trajectory
    transforms_smooth = smoothed_trajectory - trajectory
    smooth_frames = np.zeros_like(frames)
    print('\n')
    for i in range(frame_count - 1):
        print(f'\rWarping frames {i + 1}/{frame_count}',end = '')
        # Read next frame
        frame = frames[i, ...]
        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        mat = np.zeros((2, 3), np.float32)
        mat[0, 0] = np.cos(da)
        mat[0, 1] = -np.sin(da)
        mat[1, 0] = np.sin(da)
        mat[1, 1] = np.cos(da)
        mat[0, 2] = dx
        mat[1, 2] = dy
       # mat[2, 2] = 1  # Set the third row for homogenous coordinates

        # Apply homography transformation to the given frame
        #frame_stabilized = cv2.warpPerspective(frame, mat, (width, height))
        frame_stabilized = cv2.warpAffine(frame, mat, (width, height))
        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized,1.1)
        smooth_frames[i, ...] = frame_stabilized
    smooth_frames[-1, ...] = frames[-1, ...]
    return smooth_frames