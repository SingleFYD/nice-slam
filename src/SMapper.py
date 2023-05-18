import os
import time

import torch
import numpy as np
from torch.autograd import Variable
from colorama import Fore, Style

from src.common import get_camera_from_tensor, get_samples, get_tensor_from_camera, random_select
from src.utils.Sdataset import BaseDataset as SGet_dataset
from src.utils.SVisualizer import SVisualizer

class SMapper(object):
    def __init__(self, cfg, args, slam):
        self.cfg = cfg
        self.args = args 

        self.c = slam.shared_c
        self.out = slam.output
        self.render = slam.render

        self.scale = cfg['scale']

        self.device = cfg['mapping']['device']
        self.freq = cfg['mapping']['vis_freq']
        self.inside_freq = cfg['mapping']['vis_inside_freq']
        
        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = SGet_dataset(cfg, args, self.scale, self.device)
        self.n_img = len(self.frame_reader)
        self.visualizer = SVisualizer(freq = self.freq, inside_freq = self.inside_freq, vis_dir=os.path.join(self.out, 'mapping_vis'), render = self.render, device = self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

    def keyframe_selection_overlap(self, gt_depth, gt_color, c2w, key_frame_dict, k, N_samples = 16, pixel = 100):
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        device = self.device

    def optimize_map(self, num_joint_iters, lr_factor, idx, cur_gt_depth, cur_gt_color, gt_cur_c2w, keyframe_dict, keyframe_list, cur_c2w):
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        device = self.device
        cfg = self.cfg
        c = self.c
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.])).reshape([1, 4]).type(torch.float32).to(device)

    def run(self):
        cfg = self.cfg
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
        self.estimate

