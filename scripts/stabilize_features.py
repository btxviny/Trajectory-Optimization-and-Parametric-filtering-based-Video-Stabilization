import cv2
import numpy as np
import math
import torch
import argparse
from scipy.signal import medfilt as median_filter
from time import sleep
H,W = 360,640



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


def stabilize(frames,smoothing = 'mva'):
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
    smoothed_trajectory = trajectory.copy()
    if smoothing == 'mva':
        for _ in range(20):
            window_size = 15
            for idx in range(3):
                smoothed_trajectory[:, idx] = np.convolve(trajectory[:, idx], np.ones(window_size) / window_size,
                                                        mode='same')
    elif smoothing == 'ewma':
        # Apply exponential weighted moving average (EWMA) smoothing to the trajectory
        alpha = 0.7  # Smoothing factor
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
        frame_stabilized = cv2.warpPerspective(frame, M, (w, h))
        frame_stabilized = fixBorder(frame_stabilized)
        smooth_frames[i, ...] = frame_stabilized
    smooth_frames[-1, ...] = frames[-1, ...]
    return smooth_frames

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
    frames = stabilize(frames, smoothing='mva')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_path, fourcc, 30.0, (W, H))
    for idx in range(frame_count):
        img = frames[idx,...]
        out.write(img)
        cv2.imshow('window',img)
        sleep(1/30)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cv2.destroyAllWindows()
    out.release()