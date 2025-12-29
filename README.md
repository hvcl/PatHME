## PatHME: Hierarchical Multi-Expert Knowledge Distillation for Whole Slide Image Analysis 
By [Jing Wei Tan](https://scholar.google.com/citations?user=_PMI46gAAAAJ&hl=en), [Gangsu Kim](https://scholar.google.com/citations?user=CmGABBYAAAAJ&hl=en&oi=sra) and  [Won-Ki Jeong](https://scholar.google.com/citations?user=bnyKqkwAAAAJ&hl=en&oi=sra)

[Paper](https://ieeexplore.ieee.org/document/11164664)

<p align="center">
  <img src="overview.jpeg" width="1000">
</p>

### Abstract
Foundation models have shown strong generalization across a variety of pathology whole slide image (WSI) analysis tasks, including classification, segmentation and report generation. However, most existing models are trained exclusively on single-scale, high-magnification patches, often overlooking the rich contextual information available at lower magnifications. Applying such models directly to low-resolution inputs can lead to degraded or misleading feature representations, as crucial spatial context may be lost. Moreover, fine-tuning foundation models for each task is both computationally intensive and inefficient. To address these challenges, we propose Hierarchical Multi-Expert Knowledge Distillation, PatHME, a unified framework that integrates multi-scale information and leverages multiple expert foundation models for more effective and efficient WSI analysis. First, the Prompt-guided Multi-Scale Distillation module bridges the gap between low- and high-resolution representations by distilling features across magnifications using scale-aware prompts, enabling the model to capture both global tissue architecture and fine-grained cellular detail. We further introduce a Multi-Expert Knowledge Distillation strategy that enables feature-level knowledge transfer from a teacher expert to a student expert without retraining, enhancing adaptability while reducing computational cost. Finally, we incorporate a Hierarchical Multi-Scale Attention mechanism to dynamically fuse multi-resolution features based on tissue context, enabling the model to effectively capture both fine-grained details and global structural patterns. Experimental results on multiple TCGA benchmarks demonstrate that our proposed method significantly enhances model performance and efficiency, providing a scalable and generalizable solution for advanced digital pathology applications.

## Preprocessing
1. Download the raw WSI data.
2. Extract patch feature for multiple magnification (e.g: 5x and 20x).
```python
 CUDA_VISIBLE_DEVICES=0 python patchfeature_extraction.py --dir svs_directory 
```
3. Concatenate the multiscale patches from the same region to from the regional features (based on 1.25x).
```python
 python regional_feat_generation.py --task [agg_slide_feat/agg_pat_feat] --main_dir patch_feature_saving_directory --folder patch_feature_saving_folderName
```

## Training PatHME 
It is based on DINOv2. Must use at least 2 gpus and above. 
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 20510 --nproc_per_node=2 dinov2/train/train_LG.py --config-file dinov2/configs/train/kd_brca.yaml --output_dir dinov2/ckpt/pathme_test
```

## Feature Extraction from PatHME
Extract the multiscale regional feature from the pretrained PatHME model.
```python
CUDA_VISIBLE_DEVICES=1 python feature_extraction_PatHME.py --main_dir [directory of PatHME] --dataset [tcga_brca/tcga_stad/tcga_thca] --ckpt [checkpoint folder] --regFeat_dir [directory for regional feature] --iter [eg:1999]
```

## MIL
  Slide classification with attention-based MIL. Please refer to the [example label csv file](example/example_label_file.csv). The task must be same as the label column name in the label csv file.  
```python
CUDA_VISIBLE_DEVICES=0 python slide_classification_abmil.py --main_dir [/xxxx/xxx/] --dataset [tcga_brca/tcga_stad/tcga_thca] --bag_folder [folderName:gigapath/uni/virchow2/others] --feat_dim 1280 --num_epochs 200 --lr 1e-3 --dr 1e-4 --task [survival/subtype/others] 
```

## Citation
### Please cite us if you use our work. 

Tan, Jing Wei, Gangsu Kim, and Won-Ki Jeong. "PatHME: Hierarchical Multi-Expert Knowledge Distillation for Whole Slide Image Analysis." IEEE Access (2025).
