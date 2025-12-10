import sys
sys.path.append('/mnt/E/jingwei/code/dinov2_PatHME/')
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
from dinov2.fsdp import FSDPCheckpointer
#from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler
from dinov2.models import vits, vits_s2, vits_kd, vits_LG, vits_kd2

from dinov2.train.ssl_meta_arch_kd import SSLMetaArch
from tqdm import tqdm
import os, glob

import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque
import cv2
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
import pandas as pd
from tqdm import tqdm
import argparse


from concurrent.futures import ThreadPoolExecutor


seed = 0 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = 32  # Adjust as needed

def load_patch(patch_dir):
    patch = torch.tensor(np.load(patch_dir, allow_pickle=True, mmap_mode="r"))[-256:, :]
    if len(patch.shape) == 2:
        patch = patch.unsqueeze(0).permute(0, 2, 1)
    return patch




def load_patch_L1(patch_dir):
    patch_L1 = torch.tensor(np.load(patch_dir, allow_pickle=True, mmap_mode="r"))[:16, :]
    if len(patch_L1.shape) == 2:
        patch_L1 = patch_L1.unsqueeze(0)
    return patch_L1
        


def load_patch_L2(patch_dir):
    patch_L2 = torch.tensor(np.load(patch_dir, allow_pickle=True, mmap_mode="r"))[0, :]
    if len(patch_L2.shape) == 2:
        patch_L2 = patch_L2.unsqueeze(0)
    return patch_L2
        

def main(args):
    main_dir = args.main_dir
    ckpt_main_dir = f'{main_dir}/code/PatHME/dinov2/ckpt/'

    ckpt = args.ckpt




    dataset = ckpt[:4]
    folder_fn = f'{args.dataset}/gigapath_3s_agg/' 

    slide_dir = f"{args.regFeat_dir}/{folder_fn}/"
    slide_list = glob.glob(f"{slide_dir}/*") [:] 

    print ('slide number: ', len(slide_list))

    saving_dir = main_dir
    checkpoint_key = 'teacher'
    iter = args.iter


    pretrained_weights = f"{ckpt_main_dir}{ckpt}/eval/training_{iter}/{checkpoint_key}_checkpoint.pth"
    print (f'checkpoint: {pretrained_weights}')
    model = []
    if os.path.exists(pretrained_weights) == True:
        if 'Giant' in ckpt:
            print ('vit giant')
            model = vits_LG.vit_giant2(block_chunks=0, input_embed_dim=1536, output_embed_dim=1536, patch_num=256, do_hmsa=1)
        else:
            print ('vit small')
            model = vits_LG.vit_small(block_chunks=0, input_embed_dim=1536, output_embed_dim=1536, patch_num=256, do_hmsa=1)
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        print(f"Loading weights from checkpoint at {pretrained_weights}")
        state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        new_keys = []
        for key in state_dict.keys():
            if 'blocks' in key:
                parts = key.split('.')
                if len(parts) > 3 and parts[1].isdigit() and parts[2].isdigit():
                
                    new_key = '.'.join(parts[:1] + [parts[2]] + parts[3:])  
                else:
                    new_key = key 
            else:
                new_key = key 
            new_keys.append(new_key)
        updated_state_dict = {new_keys[i]: v for i, (k, v) in enumerate(state_dict.items())}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in updated_state_dict.items() if k in model_dict}
        filtered_state_dict = {k: v for k, v in model_dict.items() if k in pretrained_dict}
        model_dict.update(filtered_state_dict)
        msg = model.load_state_dict(model_dict, strict=False)
        print(f'Pretrained weights loaded from {pretrained_weights} with message: {msg}')
    else:
        print ("pretrained weight is not existing")
    model.eval()
    model.cuda()
    print ('Ckpt: ', pretrained_weights)
    feat_dir = f"{saving_dir}/{folder_fn.split('/')[0]}/"
    print (feat_dir)
    save_ms_feat_dir = f'{feat_dir}{ckpt.replace("/", "_s")}_msfeat_ep{iter}/'
    os.makedirs (save_ms_feat_dir, exist_ok=True)
    os.chmod(save_ms_feat_dir, 0o777)
    for slide_path in tqdm(slide_list):
        slide_name = os.path.split(slide_path)[1].split('.npy')[0]
        save_loc_ms = f"{save_ms_feat_dir}/{slide_name}.npy"
        if os.path.exists(save_loc_ms) == False:
            patch_list = sorted(glob.glob(f"{slide_path}/*"))
            if len(patch_list) >0 :
                slide_feat, ms_feat, slide_feat_L1 = [], [], []
                with ThreadPoolExecutor(max_workers=8) as executor:
                    patches = list(executor.map(load_patch, patch_list))
                    patches_L1 = list(executor.map(load_patch_L1, patch_list))
                    patches_L1 = torch.cat(patches_L1, dim=0).cuda()          
                patches = torch.cat(patches, dim=0).cuda()  
                for i in range(0, patches.shape[0], batch_size):
                    batch_patches = patches[i : i + batch_size]
                    batch_patches_L1 = patches_L1[i : i + batch_size]
                    with torch.no_grad():
                        output, ms_features = model.forward_features([batch_patches.view(-1, 1536, 16, 16).float(), batch_patches_L1.view(-1, 1536, 4,4).float()],masks=[None, None], do_hmsa=True ) 
                        ms_feat.append(ms_features.cpu().numpy())
                        dim = ms_features.shape[-1]
                ms_slides = np.concatenate(ms_feat, axis=0).reshape(-1, dim)
                print (save_loc_ms, ms_slides.shape)
                np.save(save_loc_ms, ms_slides)
                

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Args')

    ### Data Setting
    parser.add_argument('--main_dir', type=str, default='//')
    parser.add_argument('--dataset', type=str, default='tcga_brca')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--regFeat_dir', type=str, default='')
    parser.add_argument('--iter', type=str, default='9999')

    args = parser.parse_args()
    main(args)
