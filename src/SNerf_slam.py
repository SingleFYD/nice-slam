import os
import time

import torch
import torch.multiprocessing as mp

from src.STracker import STracker
from src.SMapper import SMapper
from src.utils.Sdecoder import SDecoder

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
        
        self.shared_decoder = SDecoder(cfg, args, self) # todo: 待修改
        

        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()

        self.tracker = STracker(cfg, args, self)
        self.mapper = SMapper(cfg, args, self)





    def tracking(self):
        while (True):
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self):
        self.mapper.run()

    def run(self):
        """
        Dispatch the processes
        """
        processes = []
        for rank in range(2):
            if rank == 0:
                p = mp.Process(target=self.tracking)
            else:
                p = mp.Process(target=self.mapping)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()




if __name__ == '__main__':
    pass