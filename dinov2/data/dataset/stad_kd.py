#from fastai.vision.all import Path, get_image_files, verify_images

from typing import Any, Optional, Callable, Tuple

from PIL import Image

from dinov2.data.datasets.extended import ExtendedVisionDataset

import numpy as np
import glob, os
import torch

class stad_kd(ExtendedVisionDataset):
    def __init__(self, dataroot, kd, transform):
        seq_list = []
        match_list = []
        #print (dataroot[:20])
        self.kd = kd
        for slide in dataroot:
            #print ('Slide: ', slide)
            feat_list = glob.glob(f"{slide}/*.npy")
            if self.kd > 0: 
                for feat_fn in feat_list:
                    slide_name = os.path.basename(os.path.dirname(feat_fn))
                    patch_coor = os.path.split(feat_fn)[-1].split('_')[1]
                    #print ('slide_name: ',slide_name, patch_coor)
                    fm_folder  = glob.glob(f'/home/Daejeon/jingwei/tcga_stad/UNI_feat_3scale_agg/{slide_name}*')[0]
                    slide_name2 = os.path.basename(fm_folder)
                    #print (fm_folder)
                    fm_feat = f'{fm_folder}/{slide_name2}_{patch_coor}'
                    if os.path.exists(fm_feat) == True:
                        #print (fm_feat, os.path.exists(fm_feat))
                        match_list.append([feat_fn,fm_feat])
                self.seq_list = match_list
            else:
                seq_list.append(feat_list)
                self.seq_list = np.concatenate (seq_list, axis=0) 
        #print (len(self.seq_list))
        self.transform = transform
        self.kd = kd
        
    def __getitem__(self, index):
        #print (self.seq_list[index])
        if self.kd > 0:
            feat = np.load(self.seq_list[index][0], allow_pickle=True)
            fm = np.load(self.seq_list[index][1], allow_pickle=True)
            seq = [feat, fm]
            #print ('fm: ', fm.shape, 'feat', feat.shape)
        else: 
            seq = np.load(self.seq_list[index], allow_pickle=True)
    
        #print (seq.shape)
        label = torch.zeros(1,1)
        #for tensor in self.transform(seq):
            #print(tensor.size())
        return self.transform(seq), label

    def __len__(self):
        return len(self.seq_list)

