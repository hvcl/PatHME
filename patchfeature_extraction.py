import os, glob, openslide, math
from openslide.deepzoom import DeepZoomGenerator
from os.path import join
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision.models import vgg16
import torch.nn as nn
import time
import random

# from datasets import WSI_set, WSI_base_set
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import csv
import traceback
import ast
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import sys


import timm
from PIL import Image
from torchvision import transforms
import torch


def resize_thumbnail_to_target(thumbnail, target_shape, orig_shape, target_size):
    foreground_coords = np.argwhere(thumbnail == 1)

    ratio_h = orig_shape[0]/target_size/thumbnail.shape[0] 
    ratio_w = orig_shape[1]/target_size/thumbnail.shape[1] 

    new_coords = []
    for coord in foreground_coords:
        ceil_row = int(math.ceil(coord[0] * ratio_h))
        floor_row = int(math.floor(coord[0] * ratio_h))
        ceil_col = int(math.ceil(coord[1] * ratio_w))
        floor_col = int(math.floor(coord[1] * ratio_w))


        for new_row in range(floor_row, min(ceil_row+1,target_shape[0])):
            for new_col in range(floor_col, min(ceil_col+1,target_shape[1])):
                # print((new_row, new_col))
                new_coords.append((new_row, new_col))

    unique_coords = list(set(new_coords))

    resized_thumbnail = np.zeros(target_shape, dtype=thumbnail.dtype)
    for row, col in unique_coords:
        resized_thumbnail[row, col] = 1

    return resized_thumbnail


def _get_grid(dzi,thumbnail_mask,thumbnail,target_downsample, fake_level, target_size, frac=1,shuffle=False, high_dzi_lv = None):
    # selected_mask = np.zeros_like(thumbnail_mask)
    if fake_level != 2:
        try:
            h,w = thumbnail_mask.shape

            for i,(tile_w,tile_h) in enumerate(dzi.level_tiles):
                if (h/tile_h)>.85 and (h/tile_h)<1.15:
                    dzi_lv = i

        except:
            dzi_lv = -1
        
    else:
        dzi_lv = int(high_dzi_lv - 4)
        target_shape = (dzi.level_tiles[dzi_lv][1], dzi.level_tiles[dzi_lv][0])
        orig_shape = (dzi.level_dimensions[dzi_lv][1], dzi.level_dimensions[dzi_lv][0])
        thumbnail_mask = resize_thumbnail_to_target(thumbnail_mask, target_shape, orig_shape, target_size)
        thumbnail = cv2.resize(thumbnail,(thumbnail_mask.shape[1],thumbnail_mask.shape[0]))
        h,w = thumbnail_mask.shape
     
    selected_mask = np.zeros_like(thumbnail_mask)
    df = pd.DataFrame(pd.DataFrame(thumbnail_mask).stack())#
    df['label'] = df[0]; df.drop(0,axis=1,inplace=True)
    
    #df['slide_path'] = self.slide_path
    df['downsample'] = target_downsample
    df['fake_level'] = fake_level
    df.query('label>0',inplace=True) # Apply when Tiling only Tissue Mask Area
   
##############################################################################
    index_list = list(df.index)
 
##############################################################################

    df['tile_loc'] = index_list
    df['rgb'] = df.apply(lambda x : _get_rgb_val(thumbnail,x['tile_loc']),axis=1)
    df.query(f'rgb>{45} & rgb<{225}',inplace=True) # Apply when Scanner Noise (Teared Slide, etc)

    df.reset_index(inplace=True,drop=True)
    
    if shuffle==True:
        df = df.sample(frac=frac).reset_index(drop=True)
    else:
        df = df[:round(frac*len(df))]

    
    tile_loc_list = [x[::-1] for x in df['tile_loc']]
   
    df['tile_loc'] = tile_loc_list
    df['dzi_lv'] = [dzi_lv]*len(df)



    for i,(w,h) in enumerate(df['tile_loc']):
        selected_mask[h,w] = df['label'].iloc[i]
    df['label'] = df['label']-1
    return df, selected_mask, dzi_lv

def _get_dzi_lv(thumbnail,dzi):
    try:
        h,w = thumbnail.shape[:2]
        for i,(tile_w,tile_h) in enumerate(dzi.level_tiles):
            if (h/tile_h)>.7 and (h/tile_h)<1.3 and (w/tile_w)>.7 and (w/tile_w)<1.3:
                return i
    except:
        return -1
        
        

def _get_property(slide, target_mpp):
    '''
    Initialize Properties(downsample, dimension) appropriate for Target MPP
    '''
    try:
        mpp = float(f'{float(slide.properties.get("openslide.mpp-x")):.2f}')
    except:
        mpp = .25
    target_downsample = int(target_mpp/mpp)     # level0: 2 / level1: 8     / level2: 32
    target_dim = tuple(x//target_downsample for x in slide.level_dimensions[0])
    
    return target_downsample, target_dim

def _get_thumbnail(slide, target_dim, dzi_size):
    '''
    Initialize Thumbnails which is appropriate for target_dim, dzi_size 
    '''
    thumbnail = np.array(
        slide.get_thumbnail(
            # (target_dim[0]//(dzi_size),target_dim[1]/(dzi_size))
            (target_dim[0]//(dzi_size),target_dim[1]//(dzi_size))
        ).convert("RGB")
    )
    thumbnail_mask = _get_tissue_mask(thumbnail)
    return thumbnail, thumbnail_mask

def _get_rgb_val(image,location):
    y_s = location[0]; x_s = location[1]
    return np.mean(image[y_s:y_s+1,x_s:x_s+1])

def _get_tissue_mask(rgb,thresh=10,morph=None,morph_kernel=(5,5)):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    r,g,b = cv2.split(rgb)
    back_r = r > cv2.threshold(r,127,255,cv2.THRESH_OTSU)[0]
    back_g = g > cv2.threshold(g,127,255,cv2.THRESH_OTSU)[0]
    back_b = b > cv2.threshold(b,127,255,cv2.THRESH_OTSU)[0]
    tissue_rgb = np.logical_not(back_r&back_g&back_b)
    tissue_s = hsv[...,1] > cv2.threshold(hsv[...,1],127,255,cv2.THRESH_OTSU)[0]

    min_r = r>127; min_g = g>127; min_b = b>127
    tissue_mask = np.array(tissue_s & (tissue_rgb + min_r + min_g + min_b)).astype(np.uint8)
    if morph=='open':
        return cv2.morphologyEx(tissue_mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(morph_kernel)))
    elif morph=='close':
        return cv2.morphologyEx(tissue_mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(morph_kernel)))
    else:
        return tissue_mask    



def expand_tissue_mask(base_mask, base_shape, target_shape):
    """
    Expand a binary tissue mask from a lower resolution level to a higher resolution level (without loops).

    Parameters:
        base_mask (np.array): Binary mask from the base level (e.g., Level 2).
        expansion_factor (int): Factor to expand the mask (e.g., 4 for Level 1, 16 for Level 0).

    Returns:
        np.array: Expanded binary mask for the target level.
    """
    # Use np.kron to expand the mask by the expansion_factor

    x = int(target_shape[0]/base_shape[0])
    y = int(target_shape[1]/base_shape[1])
            
    expanded_mask = np.kron(base_mask, np.ones((x, y), dtype=np.uint8))
    return expanded_mask


class GaussianBlur_v2(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=.5, radius_min=.1, radius_max = 2.):
        super().__init__(always_apply,p)
        self.radius_min = radius_min
        self.radius_max = radius_max
        
    def apply(self, image,**params):
        return cv2.GaussianBlur(image,(0,0),random.uniform(self.radius_min,self.radius_max))

flip_and_color_jitter_v2 = A.Compose([
    A.HorizontalFlip(p=.5),
    A.OneOf([
        A.ColorJitter(brightness=.4,contrast=.4,saturation=.2,hue=.1,p=.8),
        A.ToGray(p=.2)
    ])
]) 

normalize_v2 = A.Compose([
    A.Normalize(),ToTensorV2()  
])

global_crops_scale =(0.4,1.0)

transformation_v2 = A.Compose([
    A.RandomResizedCrop((256,256),scale=global_crops_scale,interpolation=cv2.INTER_CUBIC),
    flip_and_color_jitter_v2,
    GaussianBlur_v2(always_apply=True),
    normalize_v2
])

def pad_image(image, target_shape):
    current_shape = image.shape
    

    pad_width = [(0,max(target_shape[0]-current_shape[0])),(0,max(target_shape[1]-current_shape[1])), (0, 0)]

    padded_image = np.pad(image, pad_width, mode='constant', constant_values=255)
    
    return padded_image



class WSI_Gen(Dataset):
    def __init__(self, dzi, df, transform, preprocess):
        self.dzi = dzi
        self.df = df
        self.transform = transforms.Compose(
                            [
                                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ]
                        )
        
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.df)
    
    def pad_image(self, image, target_shape):
        current_shape = image.shape
        
        pad_width = [(0,max(target_shape[0]-current_shape[0],0)),(0,max(target_shape[1]-current_shape[1],0)), (0, 0)]

        padded_image = np.pad(image, pad_width, mode='constant', constant_values=255)
    
        return padded_image
    
    def __getitem__(self,idx):
        
        raw = self.dzi.get_tile(
                int(self.df.iloc[idx].dzi_lv),
                self.df.iloc[idx].tile_loc
            ).convert("RGB")
        
        tile = np.array(
            raw,dtype=np.uint8)

        if tile.shape[0] != 256 or tile.shape[1] !=256:
            # print(tile.shape)
            # cv2.imwrite('/workspace/240510/asdfasdf.png',tile)
            tile = self.pad_image(tile,(256,256))
            # print(tile.shape)
            # cv2.imwrite('/workspace/240510/asdfasdf_pad.png',tile)
        transformed_img = self.transform(raw)
        
        conch_img = self.preprocess(raw)
        
        
        return {
            'tiles':transformed_img, 
            'images': tile,
            'loc':np.array(self.df.iloc[idx].tile_loc),
            'downsample':np.array(self.df.iloc[idx].downsample),
            'fake_level':np.array(self.df.iloc[idx].fake_level),
            'conch_tiles':conch_img
        }
    
    
'''
patch_ex similiar to raw 
'''
def main(args):

    # Older versions of timm have compatibility issues. Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
    # tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)




    ########################################################################
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
    weights = torch.load('/mnt/wei/prov-gigapath/pytorch_model.bin')
    model.load_state_dict(weights,strict=True)
    model.eval() 
    
    model.to(args.device)
    transformation = None
    ########################################################################
    
   
    model2 = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    preprocess = create_transform(**resolve_data_config(model2.pretrained_cfg, model=model))
    model2.eval()
    model2.to(args.device)
    ########################################################################

    
    DATASET = args.dataset.upper()
    csv = glob.glob(f'{args.dir}/*.svs')
    s_pathss =  csv[:]
    os.makedirs(args.csv_dir,exist_ok=True)
    
   
    t_s = time.time()
    for i,s_paths in tqdm(enumerate(s_pathss),total=len(s_pathss)):
        try:
            s_paths = [s_paths]
 
            for s_path in s_paths:
                s_path = s_path
                s_name = s_path.split('/')[-1].split('.')[0]

                target_dir = os.path.join(args.target_dir,s_name)

                if os.path.exists (target_dir) == False: 
                    os.makedirs(target_dir,exist_ok=True)
                    
                    target_dir2 = os.path.join(args.target_dir2,s_name)
                    os.makedirs(target_dir2,exist_ok=True)


                    slide = openslide.open_slide(s_path)
                    print(slide.properties.get("openslide.mpp-x"))
                    dzi_size = int(args.target_size*(1-args.overlap))   
                    dzi = DeepZoomGenerator(slide, tile_size = dzi_size, overlap=int( (args.target_size-dzi_size)//2 ))
                    # When downsample is 4
                    tds_1,tdim_1 = _get_property(slide,args.mpp)
                    tds_2,tdim_2 = _get_property(slide,4*args.mpp)
                    tds_3,tdim_3 = _get_property(slide,16*args.mpp)
                    # tds: target_downsample level0: 2 / level1: 8     / level2: 32
                    # tdim: target_dim (tuple)      가장 고해상도 크기 // target_downsample     ex. 10000x10000 -> 200x200 
                    
                    
                    print (f"Extracting feat for {s_name}")
                    print(f'Downsample and Dimension for Each Level: {tds_1},{tds_2},{tds_3}, {tdim_1},{tdim_2},{tdim_3}')

                    thumbnail_1, _              = _get_thumbnail(slide,tdim_1,dzi_size)       
                    thumbnail_2, _              = _get_thumbnail(slide,tdim_2,dzi_size)
                    thumbnail_3,tissue_mask_3   = _get_thumbnail(slide,tdim_3,dzi_size)
                    
                
                    tissue_mask_1 = cv2.resize(tissue_mask_3, (thumbnail_1.shape[1],thumbnail_1.shape[0]), interpolation=cv2.INTER_NEAREST)
                    tissue_mask_2 = cv2.resize(tissue_mask_3, (thumbnail_2.shape[1],thumbnail_2.shape[0]), interpolation=cv2.INTER_NEAREST)

                    
                    print(f'GRID shape {thumbnail_1.shape}, {thumbnail_2.shape}, {thumbnail_3.shape}')
                    print(f'mask shape {tissue_mask_1.shape}, {tissue_mask_2.shape}, {tissue_mask_3.shape}')

                    df_1, _, dzi_lv_0 = _get_grid(dzi,tissue_mask_1,thumbnail_1,tds_1, 0, args.target_size, frac=args.sampling) 
                    df_2, _, _ = _get_grid(dzi,tissue_mask_2,thumbnail_2,tds_2, 1, args.target_size,  frac=args.sampling) 
                    df_3, _, _= _get_grid(dzi,tissue_mask_3,thumbnail_3,tds_3, 2, args.target_size,  frac=args.sampling, high_dzi_lv = dzi_lv_0)

                    df = pd.concat([df_1,df_2,df_3])
                  
                    print(f'Length of DF: {len(df_1)},{len(df_2)},{len(df_3)}')
                    wsi_gen = WSI_Gen(dzi,df,transformation,preprocess)
                    wsi_dl = DataLoader(wsi_gen, batch_size = args.batch_size,num_workers = 8,pin_memory=True, shuffle=False )
               
                
                    concat_level0 = []
                    concat_level1 = []
                    concat_level2 = []
                    
                    conch_concat_level0 = []
                    conch_concat_level1 = []
                    conch_concat_level2 = []
                    
                    for _,batch in tqdm(enumerate(wsi_dl),total=len(wsi_dl)):
                        tiles = batch['tiles']

                        images = batch['images']
                        
                        coord = batch['loc']

                        fake_level = batch['fake_level']
                        
                        conch_tiles = batch['conch_tiles']
                        
                        tiles = tiles.to(args.device,non_blocking=True)
                        feat = model.forward(tiles).cpu()
                        
                        
                        conch_tiles = conch_tiles.to(args.device,non_blocking=True)
                        
                        feat2 = model2.forward(conch_tiles).cpu()
                        
                    
                        for j,lv in enumerate(fake_level):
                            if lv == 0:
                                torch.save(
                                    feat[j].detach(),
                                    os.path.join(
                                        target_dir,
                                        f'{int(lv)}_{str(int(coord[j][0] * dzi_size)).zfill(5)}x{str(int(coord[j][1] * dzi_size)).zfill(5)}.pt'
                                    )
                                    )
                                
                                torch.save(
                                    feat2[j].detach(),
                                    os.path.join(
                                        target_dir2,
                                        f'{int(lv)}_{str(int(coord[j][0] * dzi_size)).zfill(5)}x{str(int(coord[j][1] * dzi_size)).zfill(5)}.pt'
                                    )
                                    )
                                
                                
                                
                                concat_level0.append((((int(coord[j][0] * dzi_size * 1)), int(coord[j][1] * dzi_size * 1)), feat[j].detach()))
                                
                                conch_concat_level0.append((((int(coord[j][0] * dzi_size * 1)), int(coord[j][1] * dzi_size * 1)), feat2[j].detach())) 
                                
                                
                                    
                                    
                            elif lv == 1:
                                torch.save(
                                    feat[j].detach(),                        
                                    os.path.join(
                                        target_dir,
                                        f'{int(lv)}_{str(int(coord[j][0] * dzi_size * 4)).zfill(5)}x{str(int(coord[j][1] * dzi_size * 4)).zfill(5)}.pt'
                                    )
                                    )
                                
                                
                                torch.save(
                                    feat2[j].detach(),                        
                                    os.path.join(
                                        target_dir2,
                                        f'{int(lv)}_{str(int(coord[j][0] * dzi_size * 4)).zfill(5)}x{str(int(coord[j][1] * dzi_size * 4)).zfill(5)}.pt'
                                    )
                                    )
                                
                                
                                
                                concat_level1.append((((int(coord[j][0] * dzi_size * 4)), int(coord[j][1] * dzi_size * 4)), feat[j].detach()))

                                conch_concat_level1.append((((int(coord[j][0] * dzi_size * 4)), int(coord[j][1] * dzi_size * 4)), feat2[j].detach()))
                                
                            elif lv == 2:
                                torch.save(
                                    feat[j].detach(),
                                    os.path.join(
                                        target_dir,
                                        f'{int(lv)}_{str(int(coord[j][0] * dzi_size * 16)).zfill(5)}x{str(int(coord[j][1] * dzi_size * 16)).zfill(5)}.pt'
                                    )
                                    )
                                
                                
                                torch.save(
                                    feat2[j].detach(),
                                    os.path.join(
                                        target_dir2,
                                        f'{int(lv)}_{str(int(coord[j][0] * dzi_size * 16)).zfill(5)}x{str(int(coord[j][1] * dzi_size * 16)).zfill(5)}.pt'
                                    )
                                    )
                                
                                
                                concat_level2.append((((int(coord[j][0] * dzi_size * 16)), int(coord[j][1] * dzi_size * 16)), feat[j].detach()))

                                conch_concat_level2.append((((int(coord[j][0] * dzi_size * 16)), int(coord[j][1] * dzi_size * 16)), feat2[j].detach()))
                    
                    
                    concat_level0.sort(key=lambda x: (x[0][0], x[0][1]))  # Level 0 정렬
                    concat_level1.sort(key=lambda x: (x[0][0], x[0][1]))  # Level 1 정렬
                    concat_level2.sort(key=lambda x: (x[0][0], x[0][1]))  # Level 2 정렬

                    confeat0 = torch.stack([feat for _, feat in concat_level0])
                    confeat1 = torch.stack([feat for _, feat in concat_level1])
                    confeat2 = torch.stack([feat for _, feat in concat_level2])
                    
                    torch.save(confeat0, os.path.join(target_dir,'level0_feats.pt'))
                    torch.save(confeat1, os.path.join(target_dir,'level1_feats.pt'))
                    torch.save(confeat2, os.path.join(target_dir,'level2_feats.pt'))
                    
                    
                    
                    conch_concat_level0.sort(key=lambda x: (x[0][0], x[0][1]))  # Level 0 정렬
                    conch_concat_level1.sort(key=lambda x: (x[0][0], x[0][1]))  # Level 1 정렬
                    conch_concat_level2.sort(key=lambda x: (x[0][0], x[0][1]))  # Level 2 정렬

                    conch_confeat0 = torch.stack([feat for _, feat in conch_concat_level0])
                    conch_confeat1 = torch.stack([feat for _, feat in conch_concat_level1])
                    conch_confeat2 = torch.stack([feat for _, feat in conch_concat_level2])
                    
                    torch.save(conch_confeat0, os.path.join(target_dir2,'level0_feats.pt'))
                    torch.save(conch_confeat1, os.path.join(target_dir2,'level1_feats.pt'))
                    torch.save(conch_confeat2, os.path.join(target_dir2,'level2_feats.pt'))
                    
                    with open(f"{args.csv_dir}/success_slides.csv", "a") as error_success_file:
                        error_success_file.write(s_name + "\n")
        except KeyboardInterrupt:
            print("Ctrl+C로 프로세스가 중지되었습니다.")
            break
        
        except Exception as e:
            print(f"에러 발생: {e}")
            traceback.print_exc()
            print(f"Error processing slide {s_name}")
            with open(f"{args.csv_dir}/fixed_error_slides.csv", "a") as error_error_file:
                error_error_file.write(s_name + "\n")
        
    t_e = time.time()
    print(f'Total Time: {t_e-t_s:.2f}')
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MIL Args')
    parser.add_argument('--target_size', type=int,   default=256,  help='target_size')
    parser.add_argument('--overlap',     type=float, default=0,    help='overlap')
    parser.add_argument('--sampling',    type=float, default=1,    help='sampling')
    parser.add_argument('--mpp',         type=float, default=0.5, help='mpp')
    parser.add_argument('--batch_size',  type=int,   default=8,   help='batch_size')
    parser.add_argument('--device',      type=str,   default='cuda', help='device')
    parser.add_argument('--model_name',  type=str,   default='vgg16', help='model_name')
    parser.add_argument('--dir',         type=str,   default='/workspace/I/', help='dir')
   
    
    parser.add_argument('--dataset',  type=str,   default='PIT',   help='dataset')
    
    args = parser.parse_args()
    
    args.target_dir = f'/mnt/H/gigapath/'
    args.target_dir2 = f'/mnt/H/uni/'
    args.csv_dir = f'/mnt/H/missing_patch/'

    main(args)