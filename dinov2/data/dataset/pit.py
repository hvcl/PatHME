#from fastai.vision.all import Path, get_image_files, verify_images

from typing import Any, Optional, Callable, Tuple

from PIL import Image

from dinov2.data.datasets.extended import ExtendedVisionDataset

import numpy as np
import glob
import torch

class PITDataset(ExtendedVisionDataset):
    def __init__(self, dataroot, transform):
        seq_list = []
        #print ('Dataroot: ', dataroot)
        for slide in dataroot:
            feat_list = glob.glob(f"{slide}/*.npy")
            seq_list.append(feat_list)
        self.seq_list = np.concatenate (seq_list) 
        self.transform = transform
        
    def __getitem__(self, index):
        seq = torch.tensor(np.load(self.seq_list[index], allow_pickle=True))#.permute(1, 0, 2)
        label = torch.zeros(1,1)
        if self.transform == None:
            return seq, label
        else:
            return self.transform(seq), label
    def __len__(self):
        return len(self.seq_list)


class PITDataset_s2(ExtendedVisionDataset):
    def __init__(self, dataroot, patch_num, transform):
        seq_list = dataroot        #print ('Dataroot: ', dataroot)
        self.seq_list = seq_list
        self.transform = transform
        
    def __getitem__(self, index):
        seq = torch.tensor(np.load(self.seq_list[index], allow_pickle=True))
        if len (seq.shape) == 3:
            seq = seq.permute(1, 0, 2)
        else:
            seq = seq.unsqueeze(1)
        #print ('seq: ', seq.shape)
        label = torch.zeros(1,1)
        if self.transform == None:
            return seq, label
        else:
            return self.transform(seq), label
    def __len__(self):
        return len(self.seq_list)