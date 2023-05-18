import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from src.SLogger import SLogger
# from src.SMesher import SMesher
from src.STracker import STracker
from src.SMapper import SMapper
from src.utils.SRender import SRender
from src.utils.Sdecoder import SNeEncoder
from src.utils.Sdataset import BaseDataset as SGet_dataset

mp.set_sharing_strategy('file_system')

class SNerf_slam():
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args

        self.output = args.output_folder
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.dim = cfg['data']['dim']
        self.c_dim = cfg['model']['c_dim']
        self.scale = cfg['scale']
        self.load_bound(cfg)
        # self.load_pretrain(cfg)
        self.gird_init(cfg)
        
        self.shared_decoder = SNeEncoder(self)

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = SGet_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)
        
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()
        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        # mapping of first frame is done, can begin tracking
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()
        self.mapping_cnt.share_memory_()

        self.shared_c = self.shared_c.to(self.cfg['mapping']['device']).share_memory_()
        
        self.shared_decoder = self.shared_decoder.to(self.cfg['mapping']['device'])
        self.shared_decoder.share_memory()

        self.render = SRender(cfg, args, self)
        #self.mesher = SMesher(cfg, args, self)
        self.logger = SLogger(cfg, args, self)
        self.mapper = SMapper(cfg, args, self)
        self.tracker = STracker(cfg, args, self)
        self.print_output_desc()

        
    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*self.scale)
        bound_divisible = cfg['bound_divisible']
        # enlarge the bound a bit to allow it divisible by bound_divisible
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) / bound_divisible).int()+1)*bound_divisible+self.bound[:, 0]


    def gird_init(self, cfg):
        """
        Initialize the grid for the mapping process.

        Args:
            cfg (dict): parsed config dict.

        """
        self.grid_len = cfg['grid_len']
        xyz_len = self.bound[:, 1] - self.bound[:, 0]
        c_dim = cfg['model']['c_dim']
        val_shape = list(map(int, (xyz_len/self.grid_len).tolist()))
        val_shape[0], val_shape[2] = val_shape[2], val_shape[0]
        self.val_shape = val_shape
        val_shape = [1, c_dim, *val_shape]
        val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        self.shared_c = val


    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        print(f"INFO: The GT, generated and residual depth/color images can be found under {self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")


    def tracking(self, rank):
        print("a")
        # while (True):
        #     if self.mapping_first_frame[0] == 1:
        #         break
        #     time.sleep(1)
        # self.tracker.run()


    def mapping(self, rank):
        print("b")
        # self.mapper.run()


    def run(self):
        """
        Dispatch the processes
        """
        processes = []
        for rank in range(2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank,))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == '__main__':
    pass