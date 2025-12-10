# PatHME: Hierarchical Multi-Expert Knowledge Distillation for Whole Slide Image Analysis [Paper](https://ieeexplore.ieee.org/document/11164664)

<p align="center">
  <img src="overview.jpeg" width="1000">
</p>


# Preprocessing
1. Download the raw WSI data.
2. Extract patch feature for multiple magnification (e.g: 5x and 20x).

 python 

   
4. Concatenate the multiscale patches from the same region to from the regional features (based on 1.25x).


# Training PatHME encoder.
# Download Checkpoint
The checkpoint can be downloaded from 

# Inference
   The 'best_aver.npy' should be first downlaoded from this page first.
```python
 python inference_github.py --model_path .../best.hdf5 --input_file ../xx.csv --save_path /xxx/xxx --aver_path .../best_aver.npy
```

