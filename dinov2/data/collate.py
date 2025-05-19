# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast(samples_list,kd, hmsa,  mask_ratio_tuple, mask_probability, dtype, n_tokens=273, mask_generator=None):
    # dtype = torch.half  # TODO: Remove

    if kd > 0: 
        n_fm_crops = len(samples_list[0][0]["fm_features"])
        collated_fm_crops =  torch.stack([s[0]["fm_features"][i] for i in range(n_fm_crops) for s in samples_list])
    
    if hmsa > 0:
        n_L1_crops = len(samples_list[0][0]["L1_feat"])
        n_L2_crops = len(samples_list[0][0]["L2_feat"])
        collated_L1_crops =  torch.stack([s[0]["L1_feat"][i] for i in range(n_L1_crops) for s in samples_list])
        collated_L2_crops =  torch.stack([s[0]["L2_feat"][i] for i in range(n_L2_crops) for s in samples_list])

    
    n_global_crops = len(samples_list[0][0]["global_crops"])

    
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    #print (collated_fm_crops.shape,collated_global_crops.shape, collated_local_crops.shape)

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    if kd > 0 and hmsa == 0:
        return {
        "collated_fm_crops": collated_fm_crops.to(dtype),
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        }
    elif hmsa > 0 and kd == 0:
        return {
        "collated_L1_crops": collated_L1_crops.to(dtype),
        "collated_L2_crops": collated_L2_crops.to(dtype),
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        }
    elif kd != 0 and hmsa != 0:
        return {
        "collated_fm_crops": collated_fm_crops.to(dtype),
        "collated_L1_crops": collated_L1_crops.to(dtype),
        "collated_L2_crops": collated_L2_crops.to(dtype),
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        }
    else:
        return {
            "collated_global_crops": collated_global_crops.to(dtype),
            "collated_local_crops": collated_local_crops.to(dtype),
            "collated_masks": collated_masks,
            "mask_indices_list": mask_indices_list,
            "masks_weight": masks_weight,
            "upperbound": upperbound,
            "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        }
