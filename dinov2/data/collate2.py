# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast2(samples_list,kd, hmsa,  mask_ratio_tuple, mask_probability, dtype, n_tokens=273, mask_generator=None):
    # dtype = torch.half  # TODO: Remove

    if kd > 0: 
        n_fm_crops = len(samples_list[0][0]["fm_features"])
        collated_fm_crops =  torch.stack([s[0]["fm_features"][i] for i in range(n_fm_crops) for s in samples_list])
        # n_fm_crops2 = len(samples_list[0][0]["fm_features2"])
        # collated_fm_crops2 =  torch.stack([s[0]["fm_features2"][i] for i in range(n_fm_crops2) for s in samples_list])
    
    if hmsa > 0:
        n_L1_crops = len(samples_list[0][0]["L1_feat"])
        n_L2_crops = len(samples_list[0][0]["L2_feat"])
        collated_L1_crops =  torch.stack([s[0]["L1_feat"][i] for i in range(n_L1_crops) for s in samples_list])
        collated_L2_crops =  torch.stack([s[0]["L2_feat"][i] for i in range(n_L2_crops) for s in samples_list])

    
    n_L0_crops = len(samples_list[0][0]["L0_crops"])
    n_L1_crops = len(samples_list[0][0]["L1_crops"])

    
    n_L0_local_crops = len(samples_list[0][0]["L0_local_crops"])
    n_L1_local_crops = len(samples_list[0][0]["L1_local_crops"])

    collated_L0_crops = torch.stack([s[0]["L0_crops"][i] for i in range(n_L0_crops) for s in samples_list])
    collated_L1_crops = torch.stack([s[0]["L1_crops"][i] for i in range(n_L1_crops) for s in samples_list])
    collated_L0_local_crops = torch.stack([s[0]["L0_local_crops"][i] for i in range(n_L0_local_crops) for s in samples_list])
    collated_L1_local_crops = torch.stack([s[0]["L1_local_crops"][i] for i in range(n_L1_local_crops) for s in samples_list])

    #print (collated_fm_crops.shape,collated_global_crops.shape, collated_local_crops.shape)

    B_L0 = len(collated_L0_crops)
    N_L0 = n_tokens
    n_samples_masked_L0 = int(B_L0 * mask_probability)
    probs_L0 = torch.linspace(*mask_ratio_tuple, n_samples_masked_L0 + 1)
    upperbound_L0 = 0
    masks_list_L0 = []
    mask_generator_L0 = mask_generator[0]
    for i in range(0, n_samples_masked_L0):
        prob_min_L0 = probs_L0[i]
        prob_max_L0 = probs_L0[i + 1]
        masks_list_L0.append(torch.BoolTensor(mask_generator_L0(int(N_L0 * random.uniform(prob_min_L0, prob_max_L0)))))
        upperbound_L0 += int(N_L0 * prob_max_L0)
    for i in range(n_samples_masked_L0, B_L0):
        masks_list_L0.append(torch.BoolTensor(mask_generator_L0(0)))

    random.shuffle(masks_list_L0)

    collated_masks_L0 = torch.stack(masks_list_L0).flatten(1)
    mask_indices_list_L0 = collated_masks_L0.flatten().nonzero().flatten()

    masks_weight_L0 = (1 / collated_masks_L0.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks_L0)[collated_masks_L0]

    
    B_L1 = len(collated_L1_crops)
    N_L1 = 16
    n_samples_masked_L1 = int(B_L1 * mask_probability)
    probs_L1 = torch.linspace(*mask_ratio_tuple, n_samples_masked_L1 + 1)
    upperbound_L1 = 0
    masks_list_L1 = []
    mask_generator_L1 = mask_generator[1]
    for i in range(0, n_samples_masked_L1):
        prob_min_L1 = probs_L1[i]
        prob_max_L1 = probs_L1[i + 1]
        masks_list_L1.append(torch.BoolTensor(mask_generator_L1(int(N_L1 * random.uniform(prob_min_L1, prob_max_L1)))))
        upperbound_L1 += int(N_L1 * prob_max_L1)
    for i in range(n_samples_masked_L1, B_L1):
        masks_list_L1.append(torch.BoolTensor(mask_generator_L1(0)))

    random.shuffle(masks_list_L1)

    collated_masks_L1 = torch.stack(masks_list_L1).flatten(1)
    mask_indices_list_L1 = collated_masks_L1.flatten().nonzero().flatten()

    masks_weight_L1 = (1 / collated_masks_L1.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks_L1)[collated_masks_L1]

    if kd > 0 and hmsa == 0:
        return {
        "collated_fm_crops": collated_fm_crops.to(dtype),
        #"collated_fm_crops2": collated_fm_crops2.to(dtype),
        "collated_L0_crops": collated_L0_crops.to(dtype),
        "collated_L1_crops": collated_L1_crops.to(dtype),
        "collated_L0_local_crops": collated_L0_local_crops.to(dtype),
        "collated_L1_local_crops": collated_L1_local_crops.to(dtype),
        #"collated_L2_crops" : collated_L2_crops.to(dtype),
        "collated_masks_L0": collated_masks_L0,
        "collated_masks_L1": collated_masks_L1,
        "mask_indices_list_L0": mask_indices_list_L0,
        "mask_indices_list_L1": mask_indices_list_L1,
        "masks_weight_L0": masks_weight_L0,
        "masks_weight_L1": masks_weight_L1,
        "upperbound_L0": upperbound_L0,
        "upperbound_L1": upperbound_L1,
        "n_masked_patches_L0": torch.full((1,), fill_value=mask_indices_list_L0.shape[0], dtype=torch.long),
        "n_masked_patches_L1": torch.full((1,), fill_value=mask_indices_list_L1.shape[0], dtype=torch.long),
        }
    elif hmsa > 0 and kd == 0:
        return {
        "collated_L0_crops": collated_L0_crops.to(dtype),
        "collated_L1_crops": collated_L1_crops.to(dtype),
        "collated_L0_local_crops": collated_L0_local_crops.to(dtype),
        "collated_L1_local_crops": collated_L1_local_crops.to(dtype),
        "collated_L2_crops" : collated_L2_crops.to(dtype),
        "collated_masks_L0": collated_masks_L0,
        "collated_masks_L1": collated_masks_L1,
        "mask_indices_list_L0": mask_indices_list_L0,
        "mask_indices_list_L1": mask_indices_list_L1,
        "masks_weight_L0": masks_weight_L0,
        "masks_weight_L1": masks_weight_L1,
        "upperbound_L0": upperbound_L0,
        "upperbound_L1": upperbound_L1,
        "n_masked_patches_L0": torch.full((1,), fill_value=mask_indices_list_L0.shape[0], dtype=torch.long),
        "n_masked_patches_L1": torch.full((1,), fill_value=mask_indices_list_L1.shape[0], dtype=torch.long),
        }
    elif kd != 0 and hmsa != 0:
        return {
        "collated_fm_crops": collated_fm_crops.to(dtype),
        # "collated_fm_crops2": collated_fm_crops2.to(dtype),
        "collated_L2_crops" : collated_L2_crops.to(dtype),
        "collated_L0_crops": collated_L0_crops.to(dtype),
        "collated_L1_crops": collated_L1_crops.to(dtype),
        "collated_L0_local_crops": collated_L0_local_crops.to(dtype),
        "collated_L1_local_crops": collated_L1_local_crops.to(dtype),
        "collated_masks_L0": collated_masks_L0,
        "collated_masks_L1": collated_masks_L1,
        "mask_indices_list_L0": mask_indices_list_L0,
        "mask_indices_list_L1": mask_indices_list_L1,
        "masks_weight_L0": masks_weight_L0,
        "masks_weight_L1": masks_weight_L1,
        "upperbound_L0": upperbound_L0,
        "upperbound_L1": upperbound_L1,
        "n_masked_patches_L0": torch.full((1,), fill_value=mask_indices_list_L0.shape[0], dtype=torch.long),
        "n_masked_patches_L1": torch.full((1,), fill_value=mask_indices_list_L1.shape[0], dtype=torch.long),
        }
    else:
        return {
        #"collated_fm_crops": collated_fm_crops.to(dtype),
        #"collated_L2_crops" : collated_L2_crops.to(dtype),
        "collated_L0_crops": collated_L0_crops.to(dtype),
        "collated_L1_crops": collated_L1_crops.to(dtype),
        "collated_L0_local_crops": collated_L0_local_crops.to(dtype),
        "collated_L1_local_crops": collated_L1_local_crops.to(dtype),
        "collated_masks_L0": collated_masks_L0,
        "collated_masks_L1": collated_masks_L1,
        "mask_indices_list_L0": mask_indices_list_L0,
        "mask_indices_list_L1": mask_indices_list_L1,
        "masks_weight_L0": masks_weight_L0,
        "masks_weight_L1": masks_weight_L1,
        "upperbound_L0": upperbound_L0,
        "upperbound_L1": upperbound_L1,
        "n_masked_patches_L0": torch.full((1,), fill_value=mask_indices_list_L0.shape[0], dtype=torch.long),
        "n_masked_patches_L1": torch.full((1,), fill_value=mask_indices_list_L1.shape[0], dtype=torch.long),
        }
        
