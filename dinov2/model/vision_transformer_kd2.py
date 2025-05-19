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


# Define the single prompted attention layer
class PromptedAttentionLayer_v1(nn.Module):
    def __init__(self, d_model, num_heads, num_prompts):
        super(PromptedAttentionLayer, self).__init__()
        self.d_model = d_model
        self.project_x = nn.Linear(1536, d_model)  # Project x to d_model
        self.project_y = nn.Linear(1024, d_model)  # Project y to d_model
        self.trainable_prompt = nn.Parameter(torch.randn(1, num_prompts, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)   # Adjusted for concatenated prompt

    def forward(self, x, y):
        batch_size = x.shape[0]
        # Attach (concatenate) the prompt to the query
        if x.shape[-1] != self.d_model:
            x_proj = self.project_x(x)  # (batch_size, seq_len, d_model)
        else: 
            x_proj = x
            
        y_proj = self.project_y(y)  # (batch_size, seq_len, d_model)
        prompt_expanded = self.trainable_prompt.expand(batch_size, -1, -1) # (batch_size, num_prompts, d_model)
        if x.shape[0] != y.shape[0]:
            x_proj = torch.cat([x_proj , x_proj], dim=0)
            prompt_expanded = torch.cat([prompt_expanded , prompt_expanded], dim=0)
        # Concatenate along sequence length (dim=1)
        #print (f"x: {x_proj.shape}, y:{y_proj.shape}, prompt: {prompt_expanded.shape}")
        queries = torch.cat([x_proj, prompt_expanded], dim=1)  # (batch_size, seq_len + num_prompts, feature_dim)
        #print (f"x: {queries.shape}, y:{y_proj.shape}")
        # Compute attention (query, key, value)
        attn_output, _ = self.attn(queries, y_proj, y_proj)

        return attn_output




# Single Prompted Attention Layer (Global)
class PromptedAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_prompts):
        super(PromptedAttentionLayer, self).__init__()
        self.d_model = d_model
        self.project_x = nn.Linear(1536, d_model)  # Project x to d_model
        self.project_y = nn.Linear(1024, d_model)  # Project y to d_model
        self.trainable_prompt = nn.Parameter(torch.randn(1, num_prompts, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)  # Global Attention

    def forward(self, x, y):
        batch_size = x.shape[0]

        x_proj = self.project_x(x) if x.shape[-1] != self.d_model else x
        y_proj = self.project_y(y)

        prompt_expanded = self.trainable_prompt.expand(batch_size, -1, -1)

        # Ensure batch sizes match (for uneven cases)
        if x.shape[0] != y.shape[0]:
            x_proj = torch.cat([x_proj, x_proj], dim=0)
            prompt_expanded = torch.cat([prompt_expanded, prompt_expanded], dim=0)

        # Global attention: x attends to all y
        queries = torch.cat([x_proj, prompt_expanded], dim=1)
        attn_output, _ = self.attn(queries, y_proj, y_proj)

        return attn_output

# Localized Attention Layer
class LocalizedPromptedAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_prompts):
        super(LocalizedPromptedAttentionLayer, self).__init__()
        self.d_model = d_model
        self.project_x = nn.Linear(1536, d_model)
        self.project_y = nn.Linear(1024, d_model)
        self.trainable_prompt = nn.Parameter(torch.randn(1, num_prompts, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)  # Local Attention

    def forward(self, x, y):
        batch_size = x.shape[0]

        x_proj = self.project_x(x) if x.shape[-1] != self.d_model else x
        y_proj = self.project_y(y)

        prompt_expanded = self.trainable_prompt.expand(batch_size, -1, -1)

        if x.shape[0] != y.shape[0]:
            x_proj = torch.cat([x_proj, x_proj], dim=0)
            prompt_expanded = torch.cat([prompt_expanded, prompt_expanded], dim=0)

        localized_outputs = []
        for i in range(x_proj.shape[1]):  # Iterate over each x[i]
            y_local = y_proj[:, i * 16:(i + 1) * 16, :]  # Select only 16 relevant y tokens
            queries = torch.cat([x_proj[:, i:i+1, :], prompt_expanded], dim=1)  # x[i] + prompt

            attn_output, _ = self.attn(queries, y_local, y_local)
            localized_outputs.append(attn_output)#[:, 0:1, :])  # Only take x[i]'s updated feature

        return torch.cat(localized_outputs, dim=1) 


class HMSA(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, num_prompts):
        super(HMSA, self).__init__()
        # Localized attention layers (first stage)
        self.local_layers = nn.ModuleList(
            [LocalizedPromptedAttentionLayer(d_model, num_heads, num_prompts) for _ in range(num_layers)]
        )
        # Global attention layers (second stage)
        self.global_layers = nn.ModuleList(
            [PromptedAttentionLayer(d_model, num_heads, num_prompts) for _ in range(num_layers)]
        )
    def forward(self, x, y):
        """
        x -> Features of a single patch
        y -> Features of 16 surrounding patches
        """
        # Step 1: Localized Attention
        #print ('x,y', x.shape, y.shape)
        localized_output = x
        # for l_layer in self.local_layers:
        #     localized_output = l_layer(localized_output, y)  # Localized HMSA processing

        #print ('localized_output', localized_output.shape)
        # Step 2: Global Attention
        global_output = localized_output  # Use localized features as input for global HMSA
        for g_layer in self.global_layers:
            global_output = g_layer(global_output, y)  # Global HMSA processing
        
        #print ('global_output', global_output.shape)
        
        return global_output


# Define the multi-layer prompted attention model
class HMSA_v1(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, num_prompts):
        super(HMSA, self).__init__()
        self.num_layers = num_layers
        self.attention_layers = nn.ModuleList(
            [PromptedAttentionLayer(d_model, num_heads, num_prompts) for _ in range(num_layers)]
        )

    def forward(self, x, y):
        #print (f"x: {x.shape}, y:{y.shape}")
        output = x
        for layer in self.attention_layers:
            output = layer(output, y)
        return output


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
        self.hmsa = HMSA(8, output_embed_dim , 4, 1)
        #self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = patch_num
        
        print ('num_patches: ' ,num_patches)
        self.phi = nn.Sequential(*[nn.Linear(input_embed_dim, output_embed_dim), nn.GELU(), nn.Dropout(p=0)])
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

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        #print ('npatch:', npatch, N)
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // 1
        h0 = h // 1
        #print ('w0:', w0, h0)
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        #print ('x: ', x.shape, w,h, sx,sy)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        #print ('patch_pos_embed: ', patch_pos_embed.shape)
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        #print ('patch_pos_embed:', patch_pos_embed.shape)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        if len(x.shape) == 3:
            Bnc, w, h = x.shape     
            x = x.view(-1, self.input_embed_dim, w, h)
            #print (x.shape) 
        B, nc, w, h = x.shape
        #print (B, nc, w, h)
        #x = self.patch_embed(x)
        x = x.flatten(2, 3).transpose(1,2)
        if self.input_embed_dim != self.output_embed_dim :
           x = self.phi(x)


        if masks is not None:
            #print (x.shape, masks.shape)
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
       

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

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

    def forward_features_list(self, x_list, masks_list, L1_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        
        
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        if L1_list != None:
            for x, masks, L1 in zip(all_x, masks_list, L1_list):
                x_norm = self.norm(x)
                if L1 != None:
                    hmsa_feat = self.hmsa(L1,  x_norm[:, self.num_register_tokens + 1 :])
                    #print ('hmsa_feat: ', hmsa_feat.shape ) 
                    output.append(
                        {
                        "x_norm_clstoken": x_norm[:, 0],
                        "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                        "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                        "x_prenorm": x,
                        "masks": masks,
                        "hmsa_feat": hmsa_feat,
                        })
                else:
                    output.append(
                        {
                            "x_norm_clstoken": x_norm[:, 0],
                            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                            "x_prenorm": x,
                            "masks": masks,
                        } )
        else:
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
       
        return output

    def forward_features(self, x, L1=None, L2=None,  masks=None, hmsa = False):
        
        if isinstance(x, list):
            return self.forward_features_list(x, masks, L1)

        
        x = self.prepare_tokens_with_masks(x, masks)
        
        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
  
        if L1 != None:
            #print ('doing hmsa_feat: ' )

            hmsa_feat = self.hmsa(L1,  x_norm[:, self.num_register_tokens + 1 :])
            #print ('hmsa_feat: ', hmsa_feat.shape )
            return {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                "x_prenorm": x,
                "masks": masks,
                "hmsa_feat": hmsa_feat,
            }
        else:
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
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
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
        num_heads=16,
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
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
