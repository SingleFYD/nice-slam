import os
import torch

class SLogger(object):
    """
    Save checkpoints to file

    """
    def __init__(self, cfg, args, slam):
        self.ckptsdir = slam.ckptsdir
        self.shared_c = slam.shared_c
        self.gt_c2w_list = slam.gt_c2w_list
        self.shared_decoder = slam.shared_decoder
        self.estimate_c2w_list = slam.estimate_c2w_list

    def log(self, idx, keyframe_dict, keyframe_list, selected_keyframes = None):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'c': self.shared_c,
            'decoder_state_dict': self.shared_decoder.state_dict(),
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            'keyframe_dict': keyframe_dict,
            'selected_keyframes': selected_keyframes,
            'idx': idx,
        }, path, _use_new_zipfile_serialization=False)

        print('Saved checkpoint at', path)