# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        #if not isinstance(input_size, tuple):
        #    input_size = (input_size,) * 2
        #self.height, self.width = input_size

        self.num_patches = input_size
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            #self.height,
            #self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.num_patches

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            length = int(round(target_area))
            if length < len(mask):
                start = random.randint(0, len(mask) - length)

                num_masked = mask[start:start + length].sum()
                # Overlap
                if 0 < length - num_masked <= max_mask_patches:
                    for i in range(start, start + length):
                        if mask[i] == 0:
                            mask[i] = 1
                            delta += 1

                if delta > 0:
                    break
        #print (delta)
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        #print ('masking mask: ', mask.shape)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)
            #print (f"mask: {mask.shape}, max_mask_patches: {max_mask_patches}")
            delta = self._mask(mask, max_mask_patches)
            #print ('Delta: ', delta.shape)
            if delta == 0:
                break
            else:
                mask_count += delta
        #print ('mask shape: ', mask.shape)
        #print (mask)
        return mask
