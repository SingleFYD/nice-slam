import torch
from src.common import get_rays, raw2outputs_nerf_color, sample_pdf

class SRender(object):
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
        self.points_batch_size = points_batch_size
        self.ray_batch_size = ray_batch_size

        self.lindisp = cfg['rendering']['lindisp']
        self.perturb = cfg['rendering']['perturb']
        self.N_samples = cfg['rendering']['N_samples']
        self.N_surface = cfg['rendering']['N_surface']
        self.N_importance = cfg['rendering']['N_importance']
        
        self.scale = cfg['scale']
        self.occupancy = cfg['occupancy']
        self.bound = slam.bound

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def eval_points(self, p, decoders, c=None, device='cuda:0'):
        """
        Evaluate occupancy & color of points.

        Args:
            p(tensor, N*3): point coordinates.
            decoders(nn.module): decoders.
            c(dict): feature grids.
            device(str): device to use.

        Returns:
            ret (tensor): occupancy (and color) value of input points.

        """
        p_split = torch.split(p, self.points_batch_size)
        bound = self.bound
        rets = []
        for pn in p_split:
            mask_x = (pn[:, 0] < bound[0][1]) & (pn[:, 0] > bound[0][0])
            mask_y = (pn[:, 1] < bound[1][1]) & (pn[:, 1] > bound[1][0])
            mask_z = (pn[:, 2] < bound[2][1]) & (pn[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            pn = pn.unsqueeze(0)
            ret = decoders(pn, c_grid=c).unsqueeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4: 
                ret = ret.unsqueeze(0)
            
            ret[~mask, 3] = 100
            rets.append(ret)

        ret = torch.cat(rets, dim=0)
        return ret


    def render_batch_ray(self, c, decoders, rays_o, rays_d, gt_depth=None, device='cuda:0'):
        """
        Render out depth, rgb, acc

        Args:
            c(dict): frature grids.
            decoders(nn.module): decoders.
            rays_o(tensor): origin of rays.
            rays_d(tensor): direction of rays.
            gt_depth(tensor): ground truth depth of image.
            device(str): device to use.

        Returns:
            depth(tensor, H*W): rendered depth image.
            rgb(tensor, H*W*3): rendered rgb image.
            acc(tensor, H*W): rendered acc image.

        """
        N_samples = self.N_samples
        N_surface = self.N_surface
        N_rays = rays_o.shape[0]

        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth_samples = gt_depth.repeat(1, N_samples)
        near = gt_depth_samples * 0.1

        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)
            t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

        far = torch.clamp(far_bb, 0.0, (torch.min(gt_depth) + torch.max(self.bound[:, 1] - self.bound[:, 0])) * 1.2)    # 防止far过far（背景）
        if N_surface > 0:
            gt_none_zero_mask = gt_depth > 0
            gt_none_zero = gt_depth[gt_none_zero_mask]
            gt_none_zero = gt_none_zero.unsqueeze(-1)
            gt_depth_surface = gt_none_zero.repeat(1, N_surface)
            t_vals_surface = torch.linspace(0.0, 1.0, steps=N_surface).double().to(device)

            z_vals_surface_depth_none_zero = 0.95 * gt_depth_surface * (1.0 - t_vals_surface) + 0.05 * gt_depth_surface * t_vals_surface # 范围是0.05~0.95
            z_vals_surface = torch.zeros(gt_depth.shape[0], N_surface).to(device).double()
            gt_none_zero_mask = gt_none_zero_mask.unsqueeze(-1)
            z_vals_surface[gt_none_zero_mask, :] = z_vals_surface_depth_none_zero
            near_surface = 0.001
            # 防止far_surface过大（无背景）
            if torch.max(gt_depth) - torch.min(gt_depth) < 100:
                far_surface = torch.max(gt_depth)
            else:
                far_surface = torch.max(self.bound[:, 1] - self.bound[:, 0])
            z_vals_surface_depth_zero = near_surface * (1.0 - t_vals_surface) + far_surface * t_vals_surface
            z_vals_surface_depth_zero.unsqueeze(0).repeat((~gt_none_zero_mask).sum(), 1)
            z_vals_surface[~gt_none_zero_mask, :] = z_vals_surface_depth_zero

        t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        
        if N_surface > 0:
            z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pointsf = pts.reshape(-1, 3)

        raw = self.eval_points(pointsf, decoders, c=c, device=device)
        raw = raw.reshape(N_rays, N_samples + N_surface, -1)

        depth, acc, color, _ = raw2outputs_nerf_color(raw, z_vals, rays_d, occupancy = self.occupancy, device=device)

        return depth, color, acc


    def render_img(self, c, decoders, c2w, gt_depth=None, device='cuda:0'):
        """
        Render out depth, rgb, acc
        
        Args:
            c(dict): frature grids.
            decoders(nn.module): decoders.
            c2w(tensor): camera to world matrix of current frame.
            gt_depth(tensor): ground truth depth of image.
            device(str): device to use.

        Returns:
            depth(tensor, H*W): rendered depth image.
            rgb(tensor, H*W*3): rendered rgb image.
            acc(tensor, H*W): rendered acc image.

        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy, c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            color_list = []
            acc_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                gt_depth_batch = gt_depth[i:i+ray_batch_size]
                depth, color, acc = self.render_batch_ray(c, decoders, rays_o_batch, rays_d_batch, gt_depth_batch, device)

                depth_list.append(depth)
                color_list.append(color)
                acc_list.append(acc)

            depth = torch.cat(depth_list, dim=0).reshape(H, W)
            color = torch.cat(color_list, dim=0).reshape(H, W, 3)
            acc = torch.cat(acc_list, dim=0).reshape(H, W)

            return depth, color, acc



