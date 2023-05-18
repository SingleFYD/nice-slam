import os
import time

import cv2
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
        self.bound = slam.bound
        self.logger = slam.logger
        # self.mesher = slam.mesher
        self.decoders = slam.shared_decoder
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.mapping_first_frame = slam.mapping_first_frame

        self.scale = cfg['scale']

        self.device = cfg['mapping']['device']
        self.freq = cfg['mapping']['vis_freq']
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        self.inside_freq = cfg['mapping']['vis_inside_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.fix_color = cfg['mapping']['fix_color']
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        
        self.selected_keyframes = {}
        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = SGet_dataset(cfg, args, self.scale, self.device)
        self.n_img = len(self.frame_reader)
        self.visualizer = SVisualizer(freq = self.freq, inside_freq = self.inside_freq, vis_dir=os.path.join(self.out, 'mapping_vis'), render = self.render, device = self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        """
        Frustum feature selection based on current camera pose & depth map.

        Args:
            c2w (tensor): canera pose of current frame
            key (str): name of feature grid
            val_shape (tuple): shape of grid
            depth_np (np.array): depth map of current frame

        Returns:
            mask (tensor): mask of selected optimizable featue.
            points (tensor): corresponding 3D points coordinates.

        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        X, Y, Z = torch.meshgrid(torch.linspace(self.bound[0][0], self.bound[0],[1], val_shape[2]),
                                 torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1]),
                                 torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0]))
        points = torch.stack([X, Y, Z], dim=3).reshape([-1, 3])
        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate([points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c @ homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3,3)
        cam_cord[:, 0] *= -1
        uv = K @ cam_cord
        z = uv[:, -1:] + 1e-5
        uv = uv[:, :2] /z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np, uv[i : i + remap_chunk, 0], uv[i : i + remap_chunk, 1], interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis = 0)

        edge = 0
        mask = (uv[:, 0] < W - edge) * (uv[:, 0] > edge) * (uv[:, 1] < H - edge) * (uv[:, 1] > edge)

        # 处理depth = 0空洞
        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)
        # depth test
        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths + 0.5)
        mask = mask.reshape(-1)
        
        # add feature grid near camera canter
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak - ray_o
        dist = torch.sum(dist * dist, dim=1)
        mask2 = dist < 0.5*0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]
        mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
        return mask


    def optimize_map(self, num_joint_iters, lr_factor, idx, cur_gt_depth, cur_gt_color, gt_cur_c2w, keyframe_dict, keyframe_list, cur_c2w):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enabled).

        Args:
            num_joint_iters(int): number of iterations
            lr_factor(float): the factor to times on current learning rate
            idx(int): current frame index
            cur_gt_depth(torch.Tensor): current depth map
            cur_gt_color(torch.Tensor): current color map
            gt_cur_c2w(torch.Tensor): ground truth camera to world matrix corresponding to current frame.
            keyframe_dict(list): list of selected keyframes info dictionary
            keyframe_list(list): list of selected keyframes index
            cur_c2w(torch.Tensor): estimate camera to world matrix of current frame

        Returns:
            cur_c2w(torch.Tensor): updated cur_c2w

        """

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        device = self.device
        cfg = self.cfg
        c = self.c
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.])).reshape([1, 4]).type(torch.float32).to(device)

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            num = self.mapping_window_size - 2
            optimize_frame = random_select(len(self.keyframe_list) - 1, num)

        oldest_frame = None
        if len(keyframe_dict) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list) - 1]
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]

        # save  selected keyframes info
        keyframe_info = []
        for id, frame in enumerate(optimize_frame):
            if frame == -1:
                frame_idx = keyframe_list[frame]
                tmp_gt_c2w = keyframe_dict[frame]['gt_c2w']
                tmp_est_c2w = keyframe_dict[frame]['est_c2w']
            else:
                frame_idx = idx
                tmp_gt_c2w = gt_cur_c2w
                tmp_est_c2w = cur_c2w
            keyframe_info.append({
                'idx':frame_idx,
                'gt_c2w':tmp_gt_c2w,
                'est_c2w':tmp_est_c2w,
                })
        self.selected_keyframes[idx] = keyframe_info

        pixs_per_image = self.mapping_pixels // len(optimize_frame)
            
        decoders_para_list = []
        grid_para = []
        gt_depth_np = cur_gt_depth.cpu().numpy()
        masked_c_grad = {}
        mask_c2w = cur_c2w
        for key, val in c.items():
            mask = self.get_mask_from_c2w(mask_c2w, key, val.shape[2:], gt_depth_np)
            mask = torch.from_numpy(mask).permute(2,1,0).unsqueeze(0).unsqueeze(0).repeat(1, val.shape[1], 1, 1, 1)
            val = val.to(device)
            val_grad = val[mask].clone()
            val_grad = Variable(val_grad.to(device), requires_grad=True)
            masked_c_grad[key] = val_grad
            masked_c_grad[key + 'mask'] = mask
            grid_para.append(val_grad)

        decoders_para_list += list(self.decoders.decoder.parameters())

        # BA
        cam_tensor_list = []
        gt_cam_tensor_list = []
        for frame in optimize_frame:
            # the oldest frame should be fixes to avoid drifting
            if frame != -1:
                c2w = keyframe_dict[frame]['est_c2w']
                gt_c2w = keyframe_dict[frame]['gt_c2w']
            else:
                c2w = cur_c2w
                gt_c2w = gt_cur_c2w
            cam_tensor = get_tensor_from_camera(c2w)
            cam_tensor = Variable(cam_tensor.to(device), requires_grad=True)
            cam_tensor_list.append(cam_tensor)
            gt_cam_tensor = get_tensor_from_camera(gt_c2w)
            gt_cam_tensor_list.append(gt_cam_tensor)

        # BA
        optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                        {'params': grid_para, 'lr': 0},
                                        {'params': cam_tensor_list, 'lr': 0}])
        
        for joint_iter in range(num_joint_iters):
            for key, val in c.items():
                val_grad = masked_c_grad[key]
                mask = masked_c_grad[key + 'mask']
                val = val.to(device)
                val[mask] = val_grad
                c[key] = val
            optimizer.param_groups[0]['lr'] = cfg['mapping']['decoder_lr'] * lr_factor
            optimizer.param_groups[1]['lr'] = self.BA_cam_lr

            if not (idx == 0 and self.no_vis_on_first_frame):
                self.visualizer.vis(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, self.c, self.decoders)

            optimize_frame.zero_grad()
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []

            cam_tensor_id = 0
            for frame in optimize_frame:
                if frame != -1:
                    gt_depth = keyframe_dict[frame]['depth'].to(device)
                    gt_color = keyframe_dict[frame]['color'].to(device)
                    if frame != oldest_frame:
                        cam_tensor = cam_tensor_list[cam_tensor_id]
                        cam_tensor_id += 1
                        c2w = get_camera_from_tensor(cam_tensor)
                    else:
                        c2w = keyframe_dict[frame]['est_c2w']
                else:
                    gt_depth = cur_gt_depth.to(device)
                    gt_color = cur_gt_color.to(device)
                    cam_tensor = cam_tensor_list[cam_tensor_id]
                    c2w = get_camera_from_tensor(cam_tensor)

                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
                batch_rays_o_list.append(batch_rays_o.float())
                batch_rays_d_list.append(batch_rays_d.float())
                batch_gt_depth_list.append(batch_gt_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())

            batch_rays_d = torch.cat(batch_rays_d_list)
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_gt_depth = torch.cat(batch_gt_depth_list)
            batch_gt_color = torch.cat(batch_gt_color_list)

            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)
                t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
                t, _ = torch.min(torch.max(t, dim = 2)[0], dim = 1)
                inside_mask = t >= batch_gt_depth
            ret = self.render.render_batch_ray(c, self.decoders, batch_rays_o, batch_rays_d, batch_gt_depth, device)
            depth, color, acc = ret
            depth_mask = (batch_gt_depth > 0)
            loss = torch.abs(batch_gt_depth[depth_mask] - depth[depth_mask]).sum()
            color_loss = torch.abs(batch_gt_color - color).sum()
            weighted_color_loss = self.w_color_loss * color_loss
            loss += weighted_color_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # put selected and updated features back to the grid
            for key, val in c.items():
                val_grad = masked_c_grad[key]
                mask = masked_c_grad[key + 'mask']
                val = val.detach()
                val[mask] = val_grad.clone().detach()
                c[key] = val

        # put the updates camera poses back
        cam_tensor_id = 0
        for id, frame in enumerate(optimize_frame):
            if frame != -1:
                if frame != oldest_frame:
                    c2w = get_camera_from_tensor(cam_tensor_list[cam_tensor_id].detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cam_tensor_id += 1
                    keyframe_dict[frame]['est_c2w'] = c2w.clone()
            else:
                c2w = get_camera_from_tensor(cam_tensor_list[-1].detach()) 
                c2w = torch.cat([c2w, bottom], dim=0)
                cur_c2w = c2w.clone()
        
        return cur_c2w


    def run(self):
        cfg = self.cfg
        device = self.device
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
        self.estimate_c2w_list[0] = gt_c2w.cpu()
        init = True
        prev_idx = -1
        while (1):
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img - 1:
                    break
                elif idx % self.every_frame == 0 and idx != prev_idx:
                    break
                time.sleep(0.1)
            prev_idx = idx

            print(Fore.GREEN, "Coarse Mapping Frame ",  idx.item())
            print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w = self.frame_reader[idx]

            if not init:
                lr_factor = cfg['mapping']['lr_factor']
                num_joint_iters = cfg['mapping']['iters']

                if idx == self.n_img - 1:
                    outer_joint_iters = 5
                    self.mapping_window_size *= 2
                    num_joint_iters *= 5
                    self.fix_color = True
                    self.frustum_feature_selection = False
                else:
                    outer_joint_iters = 1

            else:
                outer_joint_iters = 1
                lr_factor = cfg['mapping']['lr_first_factor']
                num_joint_iters = cfg['mapping']['iters_frist']

            cur_c2w = self.estimate_c2w_list[idx].to(device)
            num_joint_iters = num_joint_iters // outer_joint_iters
            for outer_joint_iter in range(outer_joint_iters):
                self.BA = len(self.keyframe_list) > 4 and cfg['mapping']['BA']

                if self.BA:
                    cur_c2w = self.optimize_map(num_joint_iters, lr_factor, idx, gt_depth, gt_color, gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w)
                    self.estimate_c2w_list[idx] = cur_c2w.cpu()

                # add new frame to keyframe set
                if outer_joint_iter == outer_joint_iters - 1:
                    if ((idx % self.keyframe_every == 0) or (idx == self.n_img - 2)) and (idx not in self.keyframe_list):
                        self.keyframe_list.append(idx)
                        self.keyframe_dict({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(), 'depth': gt_depth.cpu(), 'est_c2w': cur_c2w.clone()})
            torch.cuda.empty_cache()
            init = False

            # mapping of first frame is done, can begin tracking
            self.mapping_first_frame[0] = 1

            if (not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0 or idx == self.n_img - 1:
                self.logger.log(idx, self.keyframe_dict, self.keyframe_list, selected_keyframes = self.selected_keyframes )

            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1

            # if (idx % self.mesh_freq ==0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
            #     mesh_out_file = f'{self.out}/meshe/{idx:05d}_mush.ply'
            #     self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list, idx, self.device, show_forecast = self.mesh_coarse_level, clean_mesh = self.clean_mesh, get_mask_use_all_frame = False)

            if idx == self.n_img - 1:
                # mesh_out_file = f'{self.out}/meshe/final_mush.ply'
                # self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list, idx, self.device, show_forecast = self.mesh_coarse_level, clean_mesh = self.clean_mesh, get_mask_use_all_frame = False)
                # os.system(f"cp {mesh_out_file} {self.out}/meshe/{idx:05d}_mush.ply")
                break



                

            
            











