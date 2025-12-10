# PatHME: Hierarchical Multi-Expert Knowledge Distillation for Whole Slide Image Analysis 
[Paper](https://ieeexplore.ieee.org/document/11164664)

<p align="center">
  <img src="overview.jpeg" width="1000">
</p>


# Preprocessing
1. Download the raw WSI data.
2. Extract patch feature for multiple magnification (e.g: 5x and 20x).
```python
 CUDA_VISIBLE_DEVICES=0 python patchfeature_extraction.py --dir svs_directory 
```
3. Concatenate the multiscale patches from the same region to from the regional features (based on 1.25x).
```python
 python regional_feat_generation.py --task [agg_slide_feat/agg_pat_feat] --main_dir patch_feature_saving_directory --folder patch_feature_saving_folderName
```

# Training PatHME 
Must use at least 2 gpus and above. 
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 20510 --nproc_per_node=2 dinov2/train/train_LG.py --config-file dinov2/configs/train/kd_brca.yaml --output_dir dinov2/ckpt/pathme_test
```

# Feature Extraction from PatHME
```python
CUDA_VISIBLE_DEVICES=0 python
```

# MIL
  Slide classification with attention-based MIL. Please refer to the example label csv file (example_label_file.csv). The task nema must be same as the label column name in the label csv file.  
```python
CUDA_VISIBLE_DEVICES=0 python bag_classification.py --main_dir [/xxxx/xxx/] --dataset [tcga_brca/tcga_stad/tcga_thca] --bag_folder [folderName:gigapath/uni/virchow2/others] --feat_dim 1280 --num_epochs 200 --lr 1e-3 --dr 1e-4 --task [survival/subtype/others] 
```

