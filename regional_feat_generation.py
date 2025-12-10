import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import random
#import openslide
import cv2
from torchvision.models import resnet50, vgg16

def search_patch(coorX, coorY, patch_size):
    coordinates = []
    coorX = int(coorX)
    coorY = int(coorY)
    patch_size = int (patch_size)
    X = coorX*4
    Y = coorY*4
    for i in range(X, X+(patch_size*4), patch_size):
        for j in range(Y, Y+(patch_size*4), patch_size):
            coordinates.append(f"{i:05}x{j:05}")
    return coordinates




def agg_wsi(args):
    main_dir = args.main_dir
    feat_folder = args.folder 
    slide_list = glob.glob(f"{main_dir}{feat_folder}/*")
    print ('Slide#: ', len(slide_list))
    save_folder = f'{feat_folder}_s1_agg'
    os.makedirs(f"{main_dir}{save_folder}", exist_ok=True)
    print ('Generating bag --> ', f"{main_dir}{save_folder}")
    for ind in tqdm(range(len(slide_list))):
        slide_name = os.path.split(slide_list[ind])[1]
        save_loc = f"{main_dir}{save_folder}/{slide_name}.npy"
        patch_list = glob.glob(f"{slide_list[ind]}/*")
        patch_list = sorted (patch_list)
        feat = []
        for patch_path in patch_list:
            feat_  = np.load(patch_path, allow_pickle=True)
            feat.append(feat_)
        feats = np.squeeze(np.array(feat))
        print (f"{slide_list[ind]}/",'Patch#:', len(patch_list), feats.shape)
        np.save(save_loc, feats)

 
def pat(args):
    main_dir = args.main_dir
    feat_folder = args.folder 
    slide_list = glob.glob(f"{main_dir}{feat_folder}/*")
    pat_list = np.unique([os.path.split(x)[1][:17] for x in slide_list])
    save_folder = f'{feat_folder}'
    os.makedirs(f"{main_dir}{save_folder}", exist_ok=True)
    for ind in tqdm(range(len(pat_list))):
        pat_name = pat_list[ind]
        save_loc = f"{main_dir}{save_folder}/{pat_name}.npy"
        if os.path.exists(save_loc) == False:
            slide_list2 = glob.glob(f"{main_dir}{feat_folder}/{pat_name}*")
            feat = []
            for patch_path in slide_list2:
                feat_  = np.load(patch_path, allow_pickle=True)
                print ('feat_: ', feat_.shape)
                if len(feat_.shape) < 3:
                    feat_ = np.expand_dims(feat_, axis=0)
                feat.append(feat_)
            if feat_folder.split('_')[-3]=='patchToken':
                feats = np.concatenate(feat, axis=1)
            else:
                feats = np.concatenate(feat)
            print(f"#patch for {pat_name}: {len(slide_list2)} --> {feats.shape}")
            np.save(save_loc, feats)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Aggregation Args')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1200)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--task', type=str, default='feat_concat')
    parser.add_argument('--main_dir', type=str, default='/xxx/')
    parser.add_argument('--folder', type=str, default='patch_feat')
    args = parser.parse_args()

    if args.task == 'agg_slide_feat':
        agg_wsi(args)
    elif args.task == 'agg_pat_feat':
        pat(args)


        

