import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import cv2
import argparse

from time import sleep
h,w = 128,128
H,W = 720,1024
device = 'cuda'



def parse_args():
    parser = argparse.ArgumentParser(description='Video Stabilization using DMBVS-UNet')
    parser.add_argument('--in_path', type=str, help='Input video file path')
    parser.add_argument('--out_path', type=str, help='Output stabilized video file path')
    return parser.parse_args()

def show_flow(flow):
    hsv_mask = np.zeros(shape= flow.shape[:-1] +(3,),dtype = np.uint8)
    hsv_mask[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1],angleInDegrees=True)
    hsv_mask[:,:,0] = ang /2 
    hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2RGB)
    return(rgb)

def get_flow(img1,img2,raft):
    img1 = torch.from_numpy((img1 / 255.0)*2 -1).float().unsqueeze(0).permute(0,3,1,2).to(device) 
    img2 = torch.from_numpy((img2 / 255.0)*2 -1).float().unsqueeze(0).permute(0,3,1,2).to(device)
    with torch.no_grad():
        flow = raft(img1,img2)[-1]
    return flow.squeeze(0).permute(1,2,0).cpu().numpy()

def warpFlow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    warped = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return(warped)

def fixBorder(frame, cropping = 1.02):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, cropping)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def gaussian_blur(input, kernel_size, sigma):
    device  = input.device
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    y = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(x, y, indexing = 'xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / torch.sum(kernel)

    # Assuming the input is in the shape (batch_size, channels, height, width)
    channels = input.shape[1]
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, channels, 1, 1).to(device)

    padding = kernel_size // 2
    blurred = F.conv2d(input, kernel, padding=padding)
    return blurred

def get_flows(frames):
    print('Estimating Optical Flow\n')
    raft = models.optical_flow.raft_small(weights = 'Raft_Small_Weights.C_T_V2').eval().to(device)
    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    num_frames= frames.shape[0]
    flows = np.zeros((num_frames,h,w,2),dtype = np.float32)
    prev = cv2.resize(frames[0,...],(w,h))
    for idx in range(1,num_frames-1):
        curr = cv2.resize(frames[idx,...],(w,h))
        flows[idx,...] = get_flow(prev,curr,raft)
        rgb_flow = show_flow(flows[idx,...])
        prev = curr
        print(f'\ridx: {idx+1}/{num_frames}',end='')
        cv2.imshow('window',rgb_flow)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cv2.destroyAllWindows()
    del raft
    return flows

def optimize_path(flows):
    kernel_size = 15 # must be odd
    num_frames = flows.shape[0]
    pca_flows_tensor = torch.from_numpy(flows).float().permute(0, 3, 1, 2).float().to(device)
    pixel_profiles = torch.cumsum(pca_flows_tensor,dim = 0)
    smooth_profiles = pixel_profiles.clone()
    for _ in range(20):
        
        profiles_reshaped =smooth_profiles.permute(2,3,1,0).contiguous().view(-1, 2, num_frames)
        smooth_profiles = torch.nn.functional.avg_pool1d(profiles_reshaped,\
                                            kernel_size = kernel_size,\
                                            stride = 1,
                                            padding = kernel_size //2)
        smooth_profiles = smooth_profiles.permute(-1,1,0).view(num_frames,2,h,w)
    pixel_profiles = pixel_profiles.cpu()
    smooth_profiles = smooth_profiles.cpu()
    smooth_flows = smooth_profiles - pixel_profiles
    blurred_flows = gaussian_blur(smooth_flows,kernel_size=5,sigma=1).cpu()
    return blurred_flows.permute(0,2,3,1).numpy()

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
    flows = get_flows(frames)
    warps = optimize_path(flows)
    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_path, fourcc, 30.0, (W, H))
    for idx in range(frame_count):
        flow = cv2.resize(warps[idx,...],(W,H))
        img = frames[idx,...] 
        warped = warpFlow(img,flow)
        warped = fixBorder(warped,cropping=1.0)
        out.write(warped)
        cv2.imshow('window',warped)
        sleep(1/30)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cv2.destroyAllWindows()
    out.release()