import os
import time
import copy
from colorama import Fore, Style

import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.common import get_camera_from_tensor, get_samples, get_tensor_from_camera
from src.utils.Sdataset import BaseDataset as SGet_dataset
from src.utils.SVisualizer import SVisualizer

class STracker(object):
    def __init__(self, cfg, args, slam):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']

        self.bound = slam.bound
        self.shared_c = slam.shared_c
        self.output = slam.output
        self.render = slam.render
        self.mapping_idx = slam.mapping_idx
        self.gt_c2w_list = slam.gt_c2w_list
        self.shared_decoder = slam.shared_decoder
        self.estimate_c2w_list = slam.estimate_c2w_list

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']   # ignore edge pixels
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.every_frame = cfg['tracking']['every_frame']
        self.freq = cfg['tracking']['vis_freq']
        self.inside_freq = cfg['tracking']['vis_inside_freq']
        self.w_color_loss = cfg['tracking']['w_color_loss']

        self.prev_mapping_idx = -1
        
        self.frame_reader = SGet_dataset(cfg, args, self.scale, self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False, num_workers=1)

        self.visualizer = SVisualizer(freq = self.freq, inside_freq = self.inside_freq, vis_dir=os.path.join(self.output, 'tracking_vis'), render = self.render, device = self.device)
        
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
    
    def optimize_cam_in_batch(self, camera_tensor, gt_depth, gt_color, batch_size, optimizer):
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(Hedge, H - Hedge, Wedge, W - Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, device)
        
        with torch.no_grad():
            det_ray_o = batch_rays_o.clone().detach().unsqueeze(-1)
            det_ray_d = batch_rays_d.clone().detach().unsqueeze(-1)
            t = (self.bound.unsqueeze(0).to(device) - det_ray_o) / det_ray_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= batch_gt_depth
        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]

        ret = self.render.render_batch_ray(self.c, self.shared_decoder, batch_rays_o, batch_rays_d, batch_gt_depth)
        depth, color, acc = ret

        acc = acc.detach()
        tmp = torch.abs(depth - batch_gt_depth) / torch.sqrt(acc + 1e-10)
        mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        depth_loss = tmp[mask].sum()
        color_loss = torch.abs(color - batch_gt_color)[mask].sum()
        loss = depth_loss + self.w_color_loss * color_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()


    def update_para_from_mapping(self):
        if self.mapping_idx[0] != self.prev_mapping_idx:
            self.shared_decoder = copy.deepcopy(self.shared_decoder).to(self.device)
            for key, val in self.shared_c.items():
                self.c[key] = val.clone().to(self.device)
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        device = self.device
        self.c = {}
        pbar = self.frame_loader

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            idx = idx[0]
            gt_color = gt_color[0]
            gt_depth = gt_depth[0]
            gt_c2w = gt_c2w[0]

            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                while self.mapping_idx[0] != idx - 1:
                    time.sleep(0.1)
                pre_c2w = self.estimate_c2w_list[idx - 1].to(device)

            self.update_para_from_mapping()

            print(Fore.RED, "Tracking Frame ",  idx.item())
            print(Style.RESET_ALL)

            gt_camera_tensor = get_tensor_from_camera(gt_c2w)
            if idx >= 2:    # 假设匀速
                pre_c2w = pre_c2w.float()
                delta = pre_c2w @ self.estimate_c2w_list[idx - 2].to(device).float().inverse()
                estimated_new_cam_c2w = delta @ pre_c2w

            camera_tensor = get_camera_from_tensor(estimated_new_cam_c2w.detach())
            camera_tensor = Variable(camera_tensor.to(device), requires_grad=True)
            cam_para_list = [camera_tensor]
            optimizer = torch.optim.Adam(cam_para_list, lr=self.cam_lr)

            init_loss_cam_tensor = torch.abs(gt_camera_tensor.to(device) - camera_tensor).mean().item()
            candidate_cam_tensor = None
            current_min_loss = 1e10
            for cam_iter in range(self.num_cam_iters):
                self.visualizer.vis(idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.shared_decoder)

                loss = self.optimize_cam_in_batch(camera_tensor, gt_depth, gt_color, self.tracking_pixels, optimizer)

                if cam_iter == 0:
                    init_loss = loss

                loss_cam_tensor = torch.abs(gt_camera_tensor.to(device) - camera_tensor).mean().item()

                if cam_iter == self.num_cam_iters - 1:
                    print(f'Re-rendering loss: {init_loss:.2f}->{loss:.2f} ' + f'camera tensor error: {init_loss_cam_tensor:.4f}->{loss_cam_tensor:.4f}')
                
                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_cam_tensor = camera_tensor.clone().detach()

            bottom = torch.from_numpy(np.array([0, 0, 0, 1.])).reshape([1, 4]).type(torch.float32).to(device)
            c2w = get_camera_from_tensor(candidate_cam_tensor.clone().detach())
            c2w = torch.cat([c2w, bottom], dim=0)

            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            pre_c2w = c2w.clone()
            self.idx[0] = idx
            
            torch.cuda.empty_cache()   

