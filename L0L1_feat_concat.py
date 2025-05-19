import os, glob
import numpy as np

main_dir = '/home/Daejeon/jingwei/tcga_brca/'
L0_dir = 'dinov2_LG_sdkd_deepPrompt2_10prompt_L0feat_ep21999'
L1_dir = 'dinov2_LG_sdkd_deepPrompt2_10prompt_L1feat_ep21999'
ckpt, epoch = L1_dir.split('_L1feat_')
slide_list = glob.glob(f"{main_dir}{L0_dir}/*")
save_dir = f"{main_dir}{ckpt}_L0L1feat_{epoch}/"
os.makedirs(save_dir, exist_ok=True)
os.chmod(save_dir, 0o777)
for slide_path in slide_list: 
    slide_fn = os.path.split(slide_path)[1]
    L1_slide_path = f"{main_dir}{L1_dir}/{slide_fn}"
    L0_slide = np.load(slide_path)
    L1_slide = np.load(L1_slide_path)
    slide = np.concatenate ([L0_slide, L1_slide])
    print (f"L0 slide: {L0_slide.shape}, L1 slide: {L1_slide.shape}, L0L1 slide: {slide.shape}")
    loc = f"{save_dir}/{slide_fn}"
    np.save (loc, slide)
