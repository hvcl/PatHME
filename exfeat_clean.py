import os, glob
import sys
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

sys.path.append('/home/Alexandrite/jingwei/code/PIT/dinov2_PatHME/')
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
from dinov2.fsdp import FSDPCheckpointer
#from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler
from dinov2.models import vits, vits_s2

from dinov2.train.ssl_meta_arch import SSLMetaArch
from concurrent.futures import ThreadPoolExecutor



def get_vit(pretrained_weights, arch, stage='1', pat_num=0):
    if stage == '1':
        model = vits.vit_small(block_chunks=0)
    else: 
        model = vits_s2.vit_small(block_chunks=0, patchNum=pat_num)
        #print (model)
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        # Get model state_dict
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # Update model dict and load weights
        model_dict.update(pretrained_dict)
        msg = model.load_state_dict(model_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print ("Error loading pretrained model")
    return model

def get_vit(pretrained_weights, arch):
    model = vits.vit_small(block_chunks=0)
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # Update model dict and load weights
        model_dict.update(pretrained_dict)
        msg = model.load_state_dict(model_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print ("Error loading pretrained model")
    return model

##load model
stage = '1'
main_dir = '/home/Alexandrite/jingwei/'
ckpt_main_dir = f'{main_dir}/code/dinov2_PatHME/dinov2/ckpt/'
iter = '49999'
ckpt = f'5organs'
checkpoint_key = 'teacher'
pretrained_weights = f"{ckpt_main_dir}{ckpt}/eval/training_{iter}/teacher_checkpoint.pth"
arch = 'vit_small'
model = get_vit(pretrained_weights, arch)
model.eval()
model.cuda()

#print (model)
print ('Done loading pretrained model!!!')


## load vgg feature
feature_dir = '/home/Daejeon/dataset/'
folder_fn = 'vgg_feat_256_agg_full'
slide_dir = f"{feature_dir}{folder_fn}/"
print ('Slide dir:', slide_dir)
#slide_list = glob.glob(f"{slide_dir}*")[:]
print ('slide#:', len(slide_list))
slide_list = ['/home/Daejeon/dataset/vgg_feat_256_agg_full/PIT-01-MGOP-00674-IMH-002']

## generate save folder
feat_dir = f'/home/Alexandrite/jingwei/'#f"{feature_dir}/{folder_fn.split('/')[0]}/"
print (feat_dir)
save_feat_dir_patch = f'{feat_dir}{arch}_{ckpt.replace("/", "_")}_clsToken_{iter}_2/'
os.makedirs (save_feat_dir_patch, exist_ok=True)

os.chmod(save_feat_dir_patch, 0o777)
print ('save_feat_dir: ', save_feat_dir_patch)

## preprocessing the vgg features before feeding to the model
batch_size = 32  
def load_patch(patch_dir):
    patch = torch.tensor(np.load(patch_dir, allow_pickle=True, mmap_mode="r"))[:, 0, :]
    if len(patch.shape) == 2:
        patch = patch.unsqueeze(0)
    else:
        patch = patch.permute(1, 0, 2)
    return patch


for slide_path in tqdm(slide_list, desc="Processing slides"):
    slide_name = os.path.split(slide_path)[1].split('.npy')[0]
    save_loc_cls = f"{save_feat_dir_patch}/{slide_name}.npy"
    if os.path.exists(save_loc_cls):
        continue  # Skip already processed slides
    patch_list = glob.glob(f"{slide_path}/*")
    slide_feat = []
    if len(patch_list) == 0:
        continue
    with ThreadPoolExecutor(max_workers=8) as executor:
        patches = list(executor.map(load_patch, patch_list))
    patches = torch.cat(patches, dim=0).cuda()  # Move all patches to GPU as a batch
    for i in tqdm(range(0, patches.shape[0], batch_size), desc="Extracting patches"):
        batch_patches = patches[i : i + batch_size]
        with torch.no_grad():
            output = model.forward_features(batch_patches.permute(0, 2, 1))
            patch_token = output["x_norm_clstoken"].cpu().numpy()
            slide_feat.append(patch_token)
    slide_feature = np.squeeze(np.concatenate(slide_feat, axis=0))
    print(f'patch_features: {slide_feature.shape}')
    np.save(save_loc_cls, slide_feature)


if stage == '1':
    n = 0
    zero = 0
    for i in tqdm(range (len(slide_list))):
        slide_path = slide_list[i]
        slide_name = os.path.split(slide_path)[1].split('.npy')[0]#slide256_list[i]#os.path.split((os.path.split(patch_list[1])[0]))[1]
        num_patch = 273
        #print ('Extracting regional feature.')
        patch_list = glob.glob(f"{slide_path}/*")
        n = n + 1
        cls_slide = []
        pat_slide = []
        save_loc_cls = f"{save_feat_dir_cls}/{slide_name}.npy"#pth"
        save_loc_patch = f"{save_feat_dir_patch}/{slide_name}.npy"#pth"
        #print (f"{n} :Number of patch in {slide_name}: {len(patch_list)}")
        if len(patch_list) == 0:
            zero = zero + 1
        if os.path.exists(save_loc_cls) == False:
for patch_dir in patch_list:
    #if os.path.exists(save_loc_cls) == False:
            #try:
    patch = torch.tensor(np.load(patch_dir))
    #print ('original: ',patch.shape)
    if len(patch.shape)==3:
        patch = patch.permute(1, 0, 2)
    else:
        patch = patch.unsqueeze(0)
    # print ('ori: ' ,patch.shape)
    # if len(patch) >= patch_num:
    #     patch = patch[:patch_num] 
    # else:        
    #     num_repeats = patch_num // patch.size(0) + 1 
    #     print ('num_repeats: ' ,num_repeats)
    #     gcrop_1 = patch.repeat((num_repeats, 1,1))
    #     patch = gcrop_1[:patch_num]
    # print (patch.size())
    #os.makedirs (f"{save_feat_dir_cls}/{slide_name}", exist_ok=True)
    # os.makedirs (f"{save_feat_dir_patch}/{slide_name}", exist_ok=True)
    with torch.no_grad():
        output = model.forward_features(patch.permute(0,2,1).cuda())
        class_token = output["x_norm_clstoken"].cpu().numpy()
        patch_token = output["x_norm_patchtokens"].cpu().numpy()
        cls_slide.append (class_token)
        pat_slide.append (patch_token)
        #print ('patch_token: ', patch_token["x_norm_clstoken"].shape, patch_token["x_norm_patchtokens"].shape)
        
pat_slide = np.concatenate(pat_slide)
print (slide_name, cls_slide.shape, pat_slide.shape)
np.save(save_loc_cls, cls_slide)
np.save(save_loc_patch, pat_slide)
                        
        else: 
            print (f"{slide_name} --> existing!! ")
else:
    for i in tqdm(range (len(slide_list))):
        slide_path = slide_list[i]
        slide_name = os.path.split(slide_path)[1].split('.npy')[0]
        save_loc = f"{save_feat_dir_cls}/{slide_name}.npy"
        save_loc_patch = f"{save_feat_dir_patch}/{slide_name}.npy"
        if os.path.exists(save_loc)==False: 
            #num_patch = 100
            slide = torch.tensor(np.load(slide_path))
            print ('before slide: ', slide.shape)
            if len(slide.shape)==3:
               slide = slide[:,0,:]
            else:
               slide = slide.unsqueeze(0)[:,0,:]
            #print('slide: ', slide.shape)
            if len(slide.shape) < 2:
                slide = np.expand_dims(slide, axis=0)
            #ori_pat_num = len(slide)
            slide_ = slide.unsqueeze(0)
            print ('slide_:', slide_.shape)
            with torch.no_grad():
                output = model.forward_features(slide_.permute(0,2,1).cuda())
                slide_features = output["x_norm_clstoken"].cpu().numpy()
                patch_features = output["x_norm_patchtokens"].cpu().numpy()
                print ('slide_features: ', slide_features.shape, 'patch_features: ', patch_features.shape)
                #print (ori_pat_num, features.shape, token_feat.shape, token_features.shape)
                #print (slide_name, 'feature shape ' ,features.shape)
                np.save(save_loc, slide_features )
                np.save(save_loc_patch, patch_features )
        else: 
            print (f"{slide_name} --> existing!! ")
