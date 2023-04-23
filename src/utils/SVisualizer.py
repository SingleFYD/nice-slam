import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor

class SVisualizer(object):
    """
    Visualize intermediate results, render out depth, color and acc

    """
    def __init__(self, freq, inside_freq, vis_dir, render, device):
        self.freq = freq
        self.inside_freq = inside_freq
        self.vis_dir = vis_dir
        self.render = render
        self.device = device
        os.makedirs(f'{vis_dir}', exist_ok=True)


    def vis(self, idx, iter, gt_depth, gt_color, c2w, c, decoders):
        with torch.no_grad():
            if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                gt_color_np = gt_color.cpu().numpy()
                gt_depth_np = gt_depth.cpu().numpy()
                
                depth, color, _ = self.render.render_img(c, decoders, c2w, gt_depth, self.device)
                depth_np = depth.cpu().numpy()
                color_np = color.cpu().numpy()
                depth_residual = np.abs(depth_np - gt_depth_np)
                depth_residual[gt_depth_np == 0] = 0
                color_residual = np.abs(color_np - gt_color_np)
                color_residual[gt_depth_np == 0] = 0

                fig, axs = plt.subplots(2, 3)
                fig.tight_layout()
                max_depth = np.max(gt_depth_np)
                axs[0, 0].imshow(gt_depth_np, cmap='plasma', vmin=0, vmax=max_depth)
                axs[0, 0].set_title('Input depth')
                axs[0, 0].axis('off')
                axs[0, 1].imshow(depth_np, cmap='plasma', vmin=0, vmax=max_depth)
                axs[0, 1].set_title('Output depth')
                axs[0, 1].axis('off')
                axs[0, 2].imshow(depth_residual, cmap='plasma', vmin=0, vmax=max_depth)
                axs[0, 2].set_title('Depth residual')
                axs[0, 2].axis('off')

                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                axs[1, 0].imshow(gt_color_np, cmap='plasma')
                axs[1, 0].set_title('Input color')
                axs[1, 0].axis('off')
                axs[1, 1].imshow(color_np, cmap='plasma')
                axs[1, 1].set_title('Output color')
                axs[1, 1].axis('off')
                axs[1, 2].imshow(color_residual, cmap='plasma')
                axs[1, 2].set_title('Color residual')
                axs[1, 2].axis('off')

                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2)
                plt.clf()

                print(f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')