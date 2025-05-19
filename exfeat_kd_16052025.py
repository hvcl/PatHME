import sys
sys.path.append('/home/Alexandrite/jingwei/code/dinov2/')
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
#import logging

from concurrent.futures import ThreadPoolExecutor


batch_size = 32  # Adjust as needed

def load_patch(patch_dir):
    patch = torch.tensor(np.load(patch_dir, allow_pickle=True, mmap_mode="r"))[-256:, :]
    if len(patch.shape) == 2:
        patch = patch.unsqueeze(0).permute(0, 2, 1)
    return patch




def load_patch_L1(patch_dir):
    patch_L1 = torch.tensor(np.load(patch_dir, allow_pickle=True, mmap_mode="r"))[1:17, :]
    if len(patch_L1.shape) == 2:
        patch_L1 = patch_L1.unsqueeze(0)#.permute(0, 2, 1)
    return patch_L1
        


def load_patch_L2(patch_dir):
    patch_L2 = torch.tensor(np.load(patch_dir, allow_pickle=True, mmap_mode="r"))[0, :]
    if len(patch_L2.shape) == 2:
        patch_L2 = patch_L2.unsqueeze(0)#.permute(0, 2, 1)
    return patch_L2
        

main_dir = '/home/Alexandrite/jingwei/'
ckpt_main_dir = f'{main_dir}/code/dinov2/dinov2/ckpt/'

ckpt = f'stad_dinov2_kd_L0only_fold4_full_molecular'
if 'hmsa' in ckpt:
    hmsa = 1
else: 
    hmsa = 0



dataset = ckpt[:4]
folder_fn = f'tcga_{dataset}/prov_gigapath/s1_patchToken_regional/'#

slide_dir = f"/home/Daejeon/dataset/features/{folder_fn}/"

if folder_fn.split('/')[0] == 'tcga_brca':
    if 'full' in ckpt:
        print ('Full dataset')
        subset = pd.read_csv('/home/Daejeon/jingwei/tcga_brca/tcga_brca_subtype.csv')['P_ID'].to_numpy()
    else:
        subset = pd.read_csv('/home/Daejeon/jingwei/tcga_brca/tcga_brca_subtype3.csv')['P_ID'].to_numpy()
    slide_list = [f"{slide_dir}{x}" for x in subset]
    slide_list = np.concatenate([glob.glob(f"{x}*") for x in slide_list])[:]
else:
    slide_list = glob.glob(f"{slide_dir}/*")

print (len(slide_list))


checkpoint_key = 'teacher'
print ('HSMA: ', hmsa)
itera = [ '15999','17999', '19999','21999'  ]

for iter in itera:
    pretrained_weights = f"{ckpt_main_dir}{ckpt}/eval/training_{iter}/{checkpoint_key}_checkpoint.pth"
    model = []
    if os.path.exists(pretrained_weights) == True:
        #model = vits_kd.vit_small(block_chunks=0, input_embed_dim=1536, output_embed_dim=1024, patch_num=256, hmsa=0)
        model = vits_LG.vit_small(block_chunks=0, input_embed_dim=1536, output_embed_dim=1024, patch_num=256, do_hmsa=hmsa)
        #try:
        # Load the pretrained weights
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        print(f"Loading weights from checkpoint at {pretrained_weights}")
        # Extract the state dict based on the checkpoint key
        state_dict = state_dict[checkpoint_key]
        # Remove any unwanted prefixes from the state dict
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # Process the keys to remove 'blocks.x' parts
        # Process the keys to shift block indices and remove the second index
        new_keys = []
        for key in state_dict.keys():
            if 'blocks' in key:
                parts = key.split('.')
                # Check if the key matches the pattern 'blocks.x.y.something'
                if len(parts) > 3 and parts[1].isdigit() and parts[2].isdigit():
                    # Shift the block index (i.e., parts[1] becomes parts[2])
                    new_key = '.'.join(parts[:1] + [parts[2]] + parts[3:])  # parts[1] is 'x', we replace it with parts[2]
                else:
                    new_key = key  # No block index to shift, key remains unchanged
            else:
                new_key = key  # Non-blocks keys remain unchanged
            new_keys.append(new_key)
        # Create a new state_dict with updated keys
        updated_state_dict = {new_keys[i]: v for i, (k, v) in enumerate(state_dict.items())}
        # Get the model's own state_dict
        model_dict = model.state_dict()
        # Filter the updated state_dict to only include keys that are present in the model
        pretrained_dict = {k: v for k, v in updated_state_dict.items() if k in model_dict}
        # Optionally, remove extra keys in the model that are not in the pretrained state_dict
        # This ensures that any new layers in the model are not overridden
        filtered_state_dict = {k: v for k, v in model_dict.items() if k in pretrained_dict}
        # Update the model's state dict with the filtered pretrained weights
        model_dict.update(filtered_state_dict)
        # Load the updated weights into the model
        msg = model.load_state_dict(model_dict, strict=False)
        print(f'Pretrained weights loaded from {pretrained_weights} with message: {msg}')
    else:
        print ("pretrained weight is not existing")
    model.eval()
    model.cuda()
    print ('Ckpt: ', pretrained_weights)
    # save_L1_feat_dir = f'{feat_dir}{ckpt.replace("/", "_")}_L1feat_ep{iter}/'
    # os.makedirs (save_L1_feat_dir, exist_ok=True)
    # os.chmod(save_L1_feat_dir, 0o777)
    feat_dir = f"/home/Daejeon/sk_n_wei/jingwei//{folder_fn.split('/')[0]}/"
    print (feat_dir)
    if hmsa == 1:
        save_ms_feat_dir = f'{feat_dir}{ckpt.replace("/", "_s")}_msfeat_ep{iter}/'
        os.makedirs (save_ms_feat_dir, exist_ok=True)
        os.chmod(save_ms_feat_dir, 0o777)
    else:
        save_feat_dir = f'{feat_dir}{ckpt.replace("/", "_")}_L0L1feat_ep{iter}_2/'
        os.makedirs (save_feat_dir, exist_ok=True)
        os.chmod(save_feat_dir, 0o777)
        print ('save_feat_dir: ', save_feat_dir)
    for slide_path in tqdm(slide_list):
        slide_name = os.path.split(slide_path)[1].split('.npy')[0]
        #print(f"{slide_name}")
        #save_loc_L1 = f"{save_L1_feat_dir}/{slide_name}.npy"
        if hmsa == 1:
            save_loc_ms = f"{save_ms_feat_dir}/{slide_name}.npy"
        else:
            save_loc_ms = f"{save_feat_dir}/{slide_name}.npy"
        if os.path.exists(save_loc_ms) == False:# or not os.path.exists(save_loc_ms):
            patch_list = sorted(glob.glob(f"{slide_path}/*"))
            #print(f"{slide_name} --> {len(patch_list)} patches")
            if len(patch_list) >0 :
                slide_feat, ms_feat, slide_feat_L1 = [], [], []
                # Parallel patch loading
                with ThreadPoolExecutor(max_workers=8) as executor:
                    patches = list(executor.map(load_patch, patch_list))
                    patches_L1 = list(executor.map(load_patch_L1, patch_list))
                    patches_L1 = torch.cat(patches_L1, dim=0).cuda() # Move to GPU as a batc     
                    patches_L2 = list(executor.map(load_patch_L2, patch_list))
                    patches_L2 = torch.stack(patches_L2).unsqueeze(1).cuda()  # Move to GPU as a batc                 
                patches = torch.cat(patches, dim=0).cuda()  # Move to GPU as a batch
                # Process in batches
                for i in range(0, patches.shape[0], batch_size):
                    batch_patches = patches[i : i + batch_size]
                    batch_patches_L1 = patches_L1[i : i + batch_size]
                    batch_patches_L2 = patches_L2[i : i + batch_size]
                    with torch.no_grad():
                        if hmsa == 1:
                            #output, ms_features = model.forward_features([batch_patches.view(-1, 1536, 16, 16), batch_patches_L1.view(-1, 1536, 4,4)],masks=[None, None], L2_crops=batch_patches_L2, do_hmsa=True ) 
                            output, ms_features = model.forward_features([batch_patches.view(-1, 1536, 16, 16).float(), batch_patches_L1.view(-1, 1536, 4,4).float()],masks=[None, None], do_hmsa=True ) 
                            #print (batch_patches.view(-1, 1536, 16, 16).shape, batch_patches_L1.shape)
                            #ms_features = output["hmsa_feat"].cpu().numpy()
                            ms_feat.append(ms_features.cpu().numpy())
                            L0_patch_features = output[0]["x_norm_patchtokens"].cpu().numpy()
                            L1_patch_features = output[1]["x_norm_patchtokens"].cpu().numpy()
                            slide_feat.append (L0_patch_features)
                            slide_feat_L1.append (L1_patch_features)
                            dim = L1_patch_features.shape[-1]
                            #print (output["x_norm_patchtokens"].cpu().numpy().shape, ms_features.shape)
                        else:
                            output = model.forward_features([batch_patches.view(-1, 1536, 16, 16)],masks=[None], do_hmsa=False ) 
                            # output = model.forward_features([batch_patches.view(-1, 1536, 16, 16), batch_patches_L1.view(-1, 1536, 4,4)],masks=[None, None], do_hmsa=False ) 
                            L0_patch_features = output[0]["x_norm_patchtokens"].cpu().numpy()
                            # L1_patch_features = output[1]["x_norm_patchtokens"].cpu().numpy()
                            slide_feat.append (L0_patch_features)
                            # slide_feat_L1.append (L1_patch_features)
                            dim = L0_patch_features.shape[-1]
                slides = np.concatenate(slide_feat, axis=0).reshape(-1, dim)
                if hmsa == 1:
                    slides_L1 = np.concatenate(slide_feat_L1, axis=0).reshape(-1, dim)
                    ms_slides = np.concatenate(ms_feat, axis=0).reshape(-1, dim)
                    print(f"{slide_name}: {slides.shape}, {slides_L1.shape}, {ms_slides.shape}")
                    np.save(save_loc_ms, ms_slides)
                    #np.save(save_loc_L1, slides_L1)
                    #np.save(save_loc, np.concatenate([slides, slides_L1]))
                else:
                    # slides_L1 = np.concatenate(slide_feat_L1, axis=0).reshape(-1, dim)
                    print(f"{slide_name}: {slides.shape}")#, {slides_L1.shape}")
                    np.save(save_loc_ms, slides)
                    # np.save(save_loc_L1, slides_L1)
                    # np.save(save_loc_ms, np.concatenate([slides, slides_L1]))
    try:
        print (save_ms_feat_dir)
    except:
        print (save_feat_dir)
