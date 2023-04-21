import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    # 打开EXR文件
    exrfile = exr.InputFile(filename)
    # 获取图像大小
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    # 获取像素数据 sc
    ch = 'FinalImageMovieRenderQueue_WorldDepth.R'
    depth_data = exrfile.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
    depth_data = np.fromstring(depth_data, dtype=np.float32)
    depth_data = np.reshape(depth_data, isize)    # 将数据重新排列为图像格式
    return depth_data

class BaseDataset(Dataset):
    def __init__(self, cfg, args, scale, device='cuda:0'):
        super(BaseDataset, self).__init__()
        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg['cam']['png_depth_scale']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.crop_edge = cfg['cam']['crop_edge']

        self.color_paths = sorted(glob.glob(os.path.join(args.input_folder, 'color', '*.jpg')))
        self.depth_paths = sorted(glob.glob(os.path.join(args.input_folder, 'depth', '*.exr')))
        self.n_img = len(self.color_paths)

        self.poses = []
        for i in range(self.n_img):
            c2w = np.eye(4)
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

    def __len__(self):
        return self.n_img
    
    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        depth_path = self.depth_paths[idx]
        color_data = cv2.imread(color_path)
        depth_data = readEXR_onlydepth(depth_path)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB) / 255.0
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data) * self.scale
        
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        
        pose = self.poses[idx]
        pose[:3, 3] *= self.scale
        
        return idx, color_data.to(self.device), depth_data.to(self.device), pose.to(self.device)
    

        



