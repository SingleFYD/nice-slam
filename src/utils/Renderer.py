import torch
from src.common import get_rays, raw2outputs_nerf_color, sample_pdf


class Renderer(object):
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size

        self.lindisp = cfg['rendering']['lindisp']
        self.perturb = cfg['rendering']['perturb']
        self.N_samples = cfg['rendering']['N_samples']
        self.N_surface = cfg['rendering']['N_surface']
        self.N_importance = cfg['rendering']['N_importance']

        self.scale = cfg['scale']
        self.occupancy = cfg['occupancy']
        self.nice = slam.nice
        self.bound = slam.bound

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def eval_points(self, p, decoders, c=None, stage='color', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            decoders (nn.module decoders): Decoders.
            c (dicts, optional): Feature grids. Defaults to None.
            stage (str, optional): Query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): CUDA device. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        p_split = torch.split(p, self.points_batch_size)    # 将p分成多个batch_size大小的小块
        bound = self.bound
        rets = []
        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z # 标记那些点在mask外面

            pi = pi.unsqueeze(0)    # 插入一个零维度
            if self.nice:
                ret = decoders(pi, c_grid=c, stage=stage)   # decoder返回包含每个点的占据和颜色值的张量
            else:
                ret = decoders(pi, c_grid=None)
            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            ret[~mask, 3] = 100
            rets.append(ret)    # 将处理后的ret添加到rest列表中

        ret = torch.cat(rets, dim=0)
        return ret

    def render_batch_ray(self, c, decoders, rays_d, rays_o, device, stage, gt_depth=None):
        """
        Render color, depth and uncertainty of a batch of rays.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders. 
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
        """

        N_samples = self.N_samples  # 每条射线需要采样的点数 32
        N_surface = self.N_surface  # 每条射线最近的表面采样点数 16
        N_importance = self.N_importance    # 0

        N_rays = rays_o.shape[0]    # 射线数量

        if stage == 'coarse':
            gt_depth = None
        if gt_depth is None:
            N_surface = 0
            near = 0.01     # 近处深度
        else:
            gt_depth = gt_depth.reshape(-1, 1)  # 将深度数据调成一列
            gt_depth_samples = gt_depth.repeat(1, N_samples)    # 使得每个深度值都有N_sample个？
            near = gt_depth_samples*0.01    # 近处的深度值为 0.01*深度图

        with torch.no_grad():   # 不计算梯度
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # 光线原点(N, 3, 1) 克隆出来的地址不同数值相同
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # 光线方向(N, 3, 1) detach出来可以修改数值
            t = (self.bound.unsqueeze(0).to(device) - det_rays_o)/det_rays_d  # (N, 3, 2)，计算射线方向与边界相交的位置，o + t*d = bound，bound是在相机坐标系下的
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)    # (N, 1)，计算与最远边界的垂直距离
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01
           # print('far_bb', far_bb.size())

        if gt_depth is not None:
            # in case the bound is too large 将边界限制为深度图最大值的1.2倍以内
            far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))
        else:
            far = far_bb
        if N_surface > 0:   # 如果存在表面
            if False:
                # this naive implementation downgrades performance
                gt_depth_surface = gt_depth.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).to(device)
                z_vals_surface = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
            else:
                # since we want to colorize even on regions with no depth sensor readings,
                # meaning colorize on interpolated geometry region,
                # we sample all pixels (not using depth mask) for color loss.
                # Therefore, for pixels with non-zero depth value, we sample near the surface,
                # since it is not a good idea to sample 16 points near (half even behind) camera,
                # for pixels with zero depth value, we sample uniformly from camera to max_depth.
                gt_none_zero_mask = gt_depth > 0    # 获取非零深度值的掩码
                gt_none_zero = gt_depth[gt_none_zero_mask]
                gt_none_zero = gt_none_zero.unsqueeze(-1)
                gt_depth_surface = gt_none_zero.repeat(1, N_surface)
                t_vals_surface = torch.linspace(0., 1., steps=N_surface).double().to(device)
                # emperical range 0.05*depth
                z_vals_surface_depth_none_zero = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
                z_vals_surface = torch.zeros(gt_depth.shape[0], N_surface).to(device).double()
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[gt_none_zero_mask,:] = z_vals_surface_depth_none_zero
                near_surface = 0.001
                far_surface = torch.max(gt_depth)
                z_vals_surface_depth_zero = near_surface * (1.-t_vals_surface) + far_surface * (t_vals_surface)
                z_vals_surface_depth_zero.unsqueeze(0).repeat((~gt_none_zero_mask).sum(), 1)
                z_vals_surface[~gt_none_zero_mask,:] = z_vals_surface_depth_zero

        t_vals = torch.linspace(0., 1., steps=N_samples, device=device) # 在近截面和远截面之间生成N_samples个深度值

        if not self.lindisp:    # 是否对深度值进行线性采样
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        if self.perturb > 0.:   # 如果存在表面，则将表面的深度值域主采样的深度值合并，
            # get intervals(间隔) between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified(分层) samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        if N_surface > 0:
            z_vals, _ = torch.sort(
                torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3] 得到射线上采样点的坐标
        pointsf = pts.reshape(-1, 3)

        raw = self.eval_points(pointsf, decoders, c, stage, device)
        raw = raw.reshape(N_rays, N_samples+N_surface, -1)

        depth, uncertainty, color, weights = raw2outputs_nerf_color(
            raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
        if N_importance > 0:    # 是否需要对光线进行更多的采样（默认为零）
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(self.perturb == 0.), device=device)  # 原版 nerf 的分层采样
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            raw = self.eval_points(pts, decoders, c, stage, device)
            raw = raw.reshape(N_rays, N_samples+N_importance+N_surface, -1)

            depth, uncertainty, color, weights = raw2outputs_nerf_color(
                raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
            return depth, uncertainty, color

        return depth, uncertainty, color

    def render_img(self, c, decoders, c2w, device, stage, gt_depth=None):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)   # 获得射线的起点与方向
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            uncertainty_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size    # 100000
            gt_depth = gt_depth.reshape(-1) # 排成一列

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]   # ray_batch_size条射线作为一组
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=gt_depth_batch)

                depth, uncertainty, color = ret
                depth_list.append(depth.double())
                uncertainty_list.append(uncertainty.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            uncertainty = uncertainty.reshape(H, W)
            color = color.reshape(H, W, 3)
            return depth, uncertainty, color

    # this is only for imap*
    def regulation(self, c, decoders, rays_d, rays_o, gt_depth, device, stage='color'):
        """
        Regulation that discourage any geometry from the camera center to 0.85*depth.
        For imap, the geometry will not be as good if this loss is not added.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            gt_depth (tensor): sensor depth image
            device (str): device name to compute on.
            stage (str, optional):  query stage. Defaults to 'color'.

        Returns:
            sigma (tensor, N): volume density of sampled points.
        """
        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, self.N_samples)
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        near = 0.0
        far = gt_depth*0.85
        z_vals = near * (1.-t_vals) + far * (t_vals)
        perturb = 1.0
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # (N_rays, N_samples, 3)
        pointsf = pts.reshape(-1, 3)
        raw = self.eval_points(pointsf, decoders, c, stage, device)
        sigma = raw[:, -1]
        return sigma
