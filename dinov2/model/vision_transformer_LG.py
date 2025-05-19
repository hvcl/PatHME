# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


logger = logging.getLogger("dinov2")

def l2_normalize(features, eps=1e-6):
    return features / (torch.norm(features, dim=-1, keepdim=True) + eps)


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x



class LocalizedPromptedAttentionLayer2(nn.Module):
    def __init__(self, d_model, num_heads, num_prompts):
        super(LocalizedPromptedAttentionLayer2, self).__init__()
        self.d_model = d_model
        self.project_x = nn.Linear(1536, d_model)
        self.project_y = nn.Linear(1024, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.trainable_prompt = nn.Parameter(torch.zeros(1, num_prompts, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, y):
        batch_size = x.shape[0]
        x_proj = self.project_x(x) if x.shape[-1] != self.d_model else x
        y_proj = self.project_y(y)
        prompt_expanded = self.trainable_prompt.expand(batch_size, -1, -1)

        x_proj = l2_normalize(x_proj)
        y_proj = l2_normalize(y_proj)
        localized_outputs = []
        
        for i in range(x_proj.shape[1]):
            y_local = y_proj[:, i * 16:(i + 1) * 16, :]
            queries = torch.cat([x_proj, prompt_expanded], dim=1)
            attn_output, _ = self.attn(queries, y_local, y_local)#(y_local, queries, queries)#
            attn_output = self.dropout(attn_output)
            normalized_output = self.layer_norm(attn_output)
            localized_outputs.append(torch.cat([normalized_output[:, 0:1, :], prompt_expanded], dim=1))
        
        return torch.cat(localized_outputs, dim=1) #+ queries

class PromptedAttentionLayer2(nn.Module):
    def __init__(self, d_model, num_heads, num_prompts):
        super(PromptedAttentionLayer2, self).__init__()
        self.d_model = d_model
        self.project_x = nn.Linear(1536, d_model)
        self.project_y = nn.Linear(1024, d_model)
        self.trainable_prompt = nn.Parameter(torch.randn(1, num_prompts, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, y):
        batch_size = x.shape[0]
        x_proj = self.project_x(x) if x.shape[-1] != self.d_model else x
        y_proj = self.project_y(y)
        prompt_expanded = self.trainable_prompt.expand(batch_size, -1, -1)
        
        x_proj = l2_normalize(x_proj)
        y_proj = l2_normalize(y_proj)
        # if x.shape[0] != y.shape[0]:
        #     x_proj = torch.cat([x_proj, x_proj], dim=0)
        #     prompt_expanded = torch.cat([prompt_expanded, prompt_expanded], dim=0)

        queries = torch.cat([x_proj, prompt_expanded], dim=1)
        #print ('queries: ', queries.shape)
        attn_output, _ = self.attn(queries, y_proj, y_proj)
        attn_output = self.dropout(attn_output)
        normalized_output = self.layer_norm(attn_output)
        #print (normalized_output.shape)
        # Concatenate the trainable prompt instead of adding it
        updated_x = torch.cat([normalized_output[:, :x_proj.shape[1], :], prompt_expanded], dim=1)
        
        return updated_x

class LocalizedPromptedAttentionLayer_old(nn.Module):
    def __init__(self, d_model, num_heads, num_prompts):
        super(LocalizedPromptedAttentionLayer_old, self).__init__()
        self.d_model = d_model
        self.project_x = nn.Linear(1536, d_model)
        self.project_y = nn.Linear(1024, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        #self.trainable_prompt = nn.Parameter(torch.zeros(1, num_prompts, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, y):
        batch_size = x.shape[0]
        x_proj = self.project_x(x) if x.shape[-1] != self.d_model else x
        y_proj = self.project_y(y)
        #prompt_expanded = self.trainable_prompt.expand(batch_size, -1, -1)

        x_proj = l2_normalize(x_proj)
        y_proj = l2_normalize(y_proj)

        localized_outputs = []
        
        for i in range(x_proj.shape[1]):
            y_local = y_proj[:, i * 16:(i + 1) * 16, :]
            # queries = x_proj[:, i:i+1, :]
            queries = x_proj[:, i * 1:(i + 1) * 1, :]#torch.cat([, prompt_expanded], dim=1)
            attn_output, _ = self.attn(queries, y_local, y_local)#(y_local, queries, queries)#
            attn_output = self.dropout(attn_output)
            normalized_output = self.layer_norm(attn_output + queries)
            localized_outputs.append(normalized_output)#torch.cat([normalized_output[:, 0:1, :], prompt_expanded], dim=1))
        
        return torch.cat(localized_outputs, dim=1) 



class LocalizedPromptedAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_prompts):
        super(LocalizedPromptedAttentionLayer, self).__init__()
        self.d_model = d_model
        self.project_x = nn.Linear(1536, d_model)
        self.project_y = nn.Linear(1024, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        #self.trainable_prompt = nn.Parameter(torch.zeros(1, num_prompts, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y):
        if isinstance(x, list):
            x = torch.tensor(x)
        batch_size = x.shape[0]
        
        x_proj = self.project_x(x) if x.shape[-1] != self.d_model else x
        y_proj = self.project_y(y)

        x_proj = l2_normalize(x_proj)
        y_proj = l2_normalize(y_proj)

        # Split x_proj into (16 normal tokens + trainable tokens)
        x_normal = x_proj[:, -16:, :]  # [batch_size, 16, dim]
        x_trainable = x_proj[:, :10, :]  # [batch_size, num_trainable, dim]

        localized_outputs = []
        
        for i in range(x_normal.shape[1]):  # Loop over first 16 tokens
            y_local = y_proj[:, i * 16:(i + 1) * 16, :]
            queries = x_normal[:, i:i+1, :]  # Select current token
            attn_output, _ = self.attn(queries, y_local, y_local)
            attn_output = self.dropout(attn_output)
            normalized_output = self.layer_norm(attn_output + queries)
            localized_outputs.append(normalized_output)

        # # Process trainable tokens separately
        trainable_attn_output, _ = self.attn(x_trainable, y_proj, y_proj)
        trainable_attn_output = self.dropout(trainable_attn_output)
        trainable_attn_output = self.layer_norm(trainable_attn_output + x_trainable)

        # Concatenate both outputs
        return  torch.cat(localized_outputs + [trainable_attn_output], dim=1) #localized_outputs#


class PromptedAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_prompts):
        super(PromptedAttentionLayer, self).__init__()
        self.d_model = d_model
        self.project_x = nn.Linear(1536, d_model)
        self.project_y = nn.Linear(1024, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, y):
        batch_size = x.shape[0]
        x_proj = self.project_x(x) if x.shape[-1] != self.d_model else x
        y_proj = self.project_y(y)

        x_proj = l2_normalize(x_proj)
        y_proj = l2_normalize(y_proj)
        
        queries = x_proj
        #print ('queries: ', queries.shape)
        attn_output, _ = self.attn(queries, y_proj, y_proj)
        attn_output = self.dropout(attn_output)
        normalized_output = self.layer_norm(attn_output + queries)
        #print (normalized_output.shape)
        # Concatenate the trainable prompt instead of adding it
        updated_x = normalized_output#torch.cat([normalized_output[:, :x_proj.shape[1], :], prompt_expanded], dim=1)
        
        return updated_x


# HMSA Model
class HMSA(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, num_prompts):
        super(HMSA, self).__init__()
        self.local_layers = nn.ModuleList(
            [LocalizedPromptedAttentionLayer(d_model, num_heads, num_prompts) for _ in range(num_layers)]
        )
        self.global_layers = nn.ModuleList(
            [PromptedAttentionLayer(d_model, num_heads, num_prompts) for _ in range(num_layers)]
        )
    
    def forward(self, x, y):
            
        ##Split x into real input and trainable prompts
        # num_prompts = 10
        # x_main = x[:, :-num_prompts, :]  # Extract actual x (removing prompts)
        # trainable_prompts = x[:, -num_prompts:, :]  # Extract the trainable prompts

        #Pass only x_main to LocalizedPromptedAttentionLayer
        localized_output = x
        for l_layer in self.local_layers:
            localized_output = l_layer(localized_output, y)

        # # print ('localized_output before: ', localized_output.shape)
        # Concatenate back the trainable prompts
       # localized_output = torch.cat([localized_output, trainable_prompts], dim=1)

        #print ('localized_output after: ', localized_output.shape)
        # Pass the concatenated result to Global PromptedAttentionLayer
        global_output =  localized_output
        for g_layer in self.global_layers:
            global_output = g_layer(global_output, y)

        return global_output

# HMSA Model
class HMSA2(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, num_prompts):
        super(HMSA, self).__init__()
        self.local_layers = nn.ModuleList(
            [LocalizedPromptedAttentionLayer(d_model, num_heads, num_prompts) for _ in range(num_layers)]
        )
        self.global_layers = nn.ModuleList(
            [PromptedAttentionLayer(d_model, num_heads, num_prompts) for _ in range(num_layers)]
        )
    
    def forward(self, x, y):
        #print (x.shape, y.shape)
        localized_output = x
        for l_layer in self.local_layers:
            localized_output = l_layer(localized_output, y)
        
        #print (x.shape, y.shape)
        global_output = x#localized_output
        for g_layer in self.global_layers:
            global_output = g_layer(global_output, y)
        
        return global_output#, 



# HMSA Model
class HMSA_3scale(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, num_prompts):
        super(HMSA_3scale, self).__init__()
        self.local_layers_L0L1 = nn.ModuleList(
            [LocalizedPromptedAttentionLayer(d_model, num_heads, num_prompts) for _ in range(num_layers)]
        )
        self.global_layers_L0L1 = nn.ModuleList(
            [PromptedAttentionLayer(d_model, num_heads, num_prompts) for _ in range(num_layers)]
        )
        self.global_layers_L2L1 = nn.ModuleList(
            [PromptedAttentionLayer2(d_model, num_heads, num_prompts) for _ in range(num_layers)]
        )
    
    def forward(self, L2, x, y):
        localized_output = x
        for l_layer in self.local_layers_L0L1:
            localized_output = l_layer(localized_output, y)
        
        #print (x.shape, y.shape)
        global_output = localized_output
        for g_layer in self.global_layers_L0L1:
            global_output = g_layer(global_output, y)


        localized_output_L2L1 = L2
        for g_layer in self.global_layers_L2L1:
            global_output_L2L1 = g_layer(localized_output_L2L1, global_output)
        #print (global_output_L2L1.shape)
        return global_output_L2L1

class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        input_embed_dim=1536,
        output_embed_dim=1536,
        patch_num=196,
        depth=12,
        do_hmsa = 0,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        embed_dim = output_embed_dim
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.input_embed_dim = input_embed_dim
        self.output_embed_dim = output_embed_dim
        self.phi = nn.Sequential(*[nn.Linear(input_embed_dim, output_embed_dim), nn.GELU(), nn.Dropout(p=0)])
        if do_hmsa > 0 :
            print ('buiding HMSA')
            self.hmsa = HMSA(4, output_embed_dim , 4, 5) ##885
        #self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = patch_num
        

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        
        # self.prompt_tokens = nn.Parameter(torch.zeros(1,  10, embed_dim))
        #
        # self.prompt_tokens2 = nn.Parameter(torch.zeros(1, 2, embed_dim))
        #self.prompt_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Single trainable prompt per token

        # self.prompts_local = nn.Parameter(torch.randn(depth, 5, embed_dim))
        self.prompts = nn.Parameter(torch.randn(depth, 10, embed_dim))

        self.init_weights()


    def get_deep_prompts(self, blk_idx):
        """
        Get the deep prompts for a specific transformer block.
        
        Args:
            blk_idx: Index of the current block.
        
        Returns:
            A tensor of shape [num_prompts, dim].
        """
        return self.prompts[blk_idx]

    
    def get_deep_prompts_local(self, blk_idx):
        """
        Get the deep prompts for a specific transformer block.
        
        Args:
            blk_idx: Index of the current block.
        
        Returns:
            A tensor of shape [num_prompts, dim].
        """
        return self.prompts_local[blk_idx]



    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        # nn.init.normal_(self.prompt_tokens, std=1e-6)
        nn.init.normal_(self.prompts, std=1e-6)
        # nn.init.normal_(self.prompts_local, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w //1
        h0 = h //1
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        #print ('prepare_tokens_with_masks: ', x.shape, masks)
        if len(x.shape) == 3:
            Bnc, w, h = x.shape     
            x = x.view(-1, self.input_embed_dim, w, h)
        
        B, nc, w, h = x.shape
        x = x.flatten(2, 3).transpose(1,2)

        if self.input_embed_dim != self.output_embed_dim :
            x = self.phi(x)

        

        print ('x', x.shape, self.cls_token.shape)
        
    
        #
        if masks is not None:
            masks = masks.to(torch.bool)
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        # # # # # print ('before x:', x.shape)
        # if x.shape[1] == 17:
        #     x = torch.cat((x, self.prompt_tokens.expand(x.shape[0], -1, -1)), dim=1)
            #print ('after: ',x.shape)

        # if x.shape[1] == 10:
        #     x = torch.cat((x, self.prompt_tokens2.expand(x.shape[0], -1, -1)), dim=1)
            ##print ('after: ',x.shape)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list, do_hmsa, L2_crops):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        # for blk in self.blocks:
        #     x = blk(x)

        # all_x = x

        if len(x_list) == 2:
            x_high, x_low = x[0],x[1]   # x_high: [256, dim], x_low: [16, dim]
            x_high_crop, x_low_crop = None, None
        else:
            x_high, x_low, x_high_crop, x_low_crop = x[0], x[1], x[2], x[3] 
            #print (x_high.shape, x_low.shape, x_high_crop.shape, x_low_crop.shape)
        
        
        # Forward through transformer blocks with per-layer prompting (Deep Prompting only for x_low)
        for blk_idx, blk in enumerate(self.blocks):
            # Generate a new trainable prompt for this layer
            
            prompts = self.get_deep_prompts(blk_idx)  # Shape: [num_prompts, dim]
            
            num_prompts = prompts.shape[0]
            x_low_cls = x_low[:, :1, :] 

            if blk_idx == 0:
                x_low_token = x_low[:, 1:, :]
            else:
                x_low_tokens = x_low[:, 1 + num_prompts:, :]
            x_low =  torch.cat([x_low_cls, prompts.expand(x_low.shape[0], -1, -1), x_low_token], dim=1)
            #torch.cat([ x_low, prompts.expand(x_low.shape[0], -1, -1)], dim=1)
            #print (blk_idx, 'x_low', x_low.shape)
            # Process both scales separately
            x_low = blk(x_low)  # x_low gets deep prompting
            x_high = blk(x_high)  # x_high remains unchanged
            # Process x_high if x_high_crop and x_low_crop exist
            if x_high_crop is not None and x_low_crop is not None:
                #prompts_local = self.get_deep_prompts(blk_idx)
                # Process crops with the transformer block
                x_high_crop = blk(x_high_crop)  # Process x_high_crop with the block
                #x_low_crop_cls = x_low_crop[:, :1, :] 
                #if blk_idx == 0:
                #    x_low_crop_token = x_low_crop[:, 1:, :]
                #else:
                #    x_low_crop_token = x_low_crop[:, 1 + num_prompts:, :]    
                #x_low_crop =  torch.cat([x_low_crop_cls, prompts.expand(x_low_crop.shape[0], -1, -1), x_low_crop_token], dim=1)
                x_low_crop = blk(x_low_crop)    # Process x_low_crop with the block


        if x_high_crop is not None and x_low_crop is not None:
            all_x = [x_high, x_low, x_high_crop, x_low_crop]
        else:
            all_x = [x_high, x_low]  # Reversed order for output
        
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        if do_hmsa == True:
            if L2_crops != None:
                #print (L2_crops.shape, all_x[1][:,1:,].shape, all_x[0][:,1:,].shape)
                hmsa_feat = self.hmsa(L2_crops, all_x[1][:,1:,], all_x[0][:,1:,]) 
            else:
                hmsa_feat = self.hmsa(all_x[1][:,1:,], all_x[0][:,1:,])
            return output, hmsa_feat
        else:
            return output

    def forward_features(self, x, masks=[None, None],L2_crops=None,  do_hmsa=True):
        if isinstance(x, list):
            #print ('forward_features: ', len(x), masks)
            return self.forward_features_list(x, masks, do_hmsa,L2_crops)

        x = self.prepare_tokens_with_masks(x, masks)

        # for blk in self.blocks:
        #     x = blk(x)


        # for layer_idx, blk in enumerate(self.blocks):
        #     prompts = self.deep_prompts[layer_idx].expand(x.shape[0], -1, -1)
        #     x = blk(torch.cat([prompts, x], dim=1))

    
        
        # Forward through transformer blocks with per-layer prompting (Deep Prompting only for x_low)
        for blk_idx, blk in enumerate(self.blocks):
            # Generate a new trainable prompt for this layer
            if x.shape[1] == 17:
                prompts = self.get_deep_prompts(blk_idx)  # Shape: [num_prompts, dim]
                
                num_prompts = prompts.shape[0]
                x_cls = x[:, :1, :] 

                if blk_idx == 0:
                    x_token = x[:, 1:, :]
                else:
                    x_tokens = x[:, 1 + num_prompts:, :]
                x =  torch.cat([x_cls, prompts.expand(x.shape[0], -1, -1), x_token], dim=1)
            else:
                x = x
            x = blk(x)  # x_low gets deep prompting


        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model