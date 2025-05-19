# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss#, iBOTPatchLoss_L0L1
from dinov2.models import build_model_from_cfg_LG
from dinov2.layers import DINOHead, Mlp
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model
from fvcore.nn import FlopCountAnalysis

from dinov2.models.vision_transformer import BlockChunk

import torch.nn.functional as F

# try:
#     from xformers.ops import fmha
# except ImportError:
#     raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg_LG(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head
        self.do_kd = cfg.kd.loss_weight > 0
        self.do_hmsa = cfg.hmsa.loss_weight > 0

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_kd:
            self.mlp = Mlp(in_features=1024, hidden_features=1024,  out_features=512)
            self.mlp2 = Mlp(in_features=512, hidden_features=512,  out_features=512)

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            #self.ibot_patch_loss_L0L1 = iBOTPatchLoss_L0L1(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        
        if self.do_kd:
            self.kd_loss_weight = cfg.kd.loss_weight
        
        
        if self.do_hmsa:
            self.hmsa_loss_weight = cfg.hmsa.loss_weight
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def forward_backward(self, images, teacher_temp):
        n_global_crops = 1
        #assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        L0_crops = images["collated_L0_crops"].cuda(non_blocking=True)
        L1_crops = images["collated_L1_crops"].cuda(non_blocking=True)
        
        L0_local_crops = images["collated_L0_local_crops"].cuda(non_blocking=True)
        L1_local_crops = images["collated_L1_local_crops"].cuda(non_blocking=True)


        masks_L0 = images["collated_masks_L0"].cuda(non_blocking=True)
        mask_indices_list_L0 = images["mask_indices_list_L0"].cuda(non_blocking=True)
        n_masked_patches_tensor_L0 = images["n_masked_patches_L0"].cuda(non_blocking=True)
        n_masked_patches_L0 = mask_indices_list_L0.shape[0]
        upperbound_L0 = images["upperbound_L0"]
        masks_weight_L0 = images["masks_weight_L0"].cuda(non_blocking=True)

        
        masks_L1 = images["collated_masks_L1"].cuda(non_blocking=True)
        mask_indices_list_L1 = images["mask_indices_list_L1"].cuda(non_blocking=True)
        n_masked_patches_tensor_L1 = images["n_masked_patches_L1"].cuda(non_blocking=True)
        n_masked_patches_L1 = mask_indices_list_L1.shape[0]
        upperbound_L1 = images["upperbound_L1"]
        masks_weight_L1 = images["masks_weight_L1"].cuda(non_blocking=True)


        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = 1#(n_global_crops - 1) * n_global_crops
        #print ('n_local_crops_loss_terms:', n_local_crops_loss_terms, 'n_global_crops_loss_terms', n_global_crops_loss_terms)

        do_dino = self.do_dino
        do_ibot = self.do_ibot
        do_kd = self.do_kd
        do_hmsa = self.do_hmsa

        if do_hmsa:
            L2_crops = images["collated_L2_crops"].cuda(non_blocking=True)
        #print ('do_hmsa: ', do_hmsa)
        
        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops
        
       
      
        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x_L0, n_L0_crops_teacher = L0_crops, n_global_crops
            x_L1, n_L1_crops_teacher = L1_crops, n_global_crops

            model =  self.teacher.backbone
            from torch.profiler import profile, ProfilerActivity
            model.eval()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_flops=True,record_shapes=True, record_shapes=True) as prof:
                model([x_L0, x_L1],masks=[None, None], do_hmsa = do_hmsa, is_training=True)
            
            print(prof.key_averages().table(sort_by="flops", row_limit=10))
            #print ('x:', x_L0.shape, x_L1.shape)
            if do_hmsa:
                # teacher_backbone_output_dict, teacher_hmsa = self.teacher.backbone([x_L0, x_L1],masks=[None, None],L2_crops= L2_crops, do_hmsa = do_hmsa, is_training=True)    
                teacher_backbone_output_dict, teacher_hmsa = self.teacher.backbone([x_L0, x_L1],masks=[None, None], do_hmsa = do_hmsa, is_training=True)    
            else:
                teacher_backbone_output_dict = self.teacher.backbone([x_L0, x_L1],masks=[None, None], do_hmsa = do_hmsa, is_training=True)
            from ptflops import get_model_complexity_info

            def input_constructor(input_res):
                # Create dummy inputs with specified resolutions
                x_L0 = torch.randn(1, 3, *input_res[0])
                x_L1 = torch.randn(1, 3, *input_res[1])
                return dict(input=(x_L0, x_L1))

            # Define the input resolutions for x_L0 and x_L1
            input_resolutions = [(256,156), (16,1536)]

            # Calculate FLOPs and parameters
            macs, params = get_model_complexity_info(
                model,
                input_res=input_resolutions,
                input_constructor=input_constructor,
                as_strings=True,
                print_per_layer_stat=True
            )
            print(f"FLOPs: {macs}, Parameters: {params}")


            teacher_cls_tokens_L0 = teacher_backbone_output_dict[0]["x_norm_clstoken"]
            teacher_cls_tokens_L1 = teacher_backbone_output_dict[1]["x_norm_clstoken"]
    
            ibot_teacher_patch_tokens_L0 = teacher_backbone_output_dict[0]["x_norm_patchtokens"]
            ibot_teacher_patch_tokens_L1 = teacher_backbone_output_dict[1]["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens_L0.shape[-1]
            n_cls_tokens = teacher_cls_tokens_L0.shape[0]
            #print ('_dim: ', _dim, n_cls_tokens)

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher_L0 = ibot_teacher_patch_tokens_L0.new_zeros(upperbound_L0 + n_cls_tokens, _dim)
                buffer_tensor_teacher_L0[:n_cls_tokens].copy_(teacher_cls_tokens_L0)
                torch.index_select(
                    ibot_teacher_patch_tokens_L0.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list_L0,
                    out=buffer_tensor_teacher_L0[n_cls_tokens : n_cls_tokens + n_masked_patches_L0],
                )
                tokens_after_head_L0 = self.teacher.dino_head(buffer_tensor_teacher_L0)
                teacher_cls_tokens_after_head_L0 = tokens_after_head_L0[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head_L0 = tokens_after_head_L0[
                    n_cls_tokens : n_cls_tokens + n_masked_patches_L0
                ]
                
                buffer_tensor_teacher_L1 = ibot_teacher_patch_tokens_L1.new_zeros(upperbound_L1 + n_cls_tokens, _dim)
                buffer_tensor_teacher_L1[:n_cls_tokens].copy_(teacher_cls_tokens_L1)
                torch.index_select(
                    ibot_teacher_patch_tokens_L1.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list_L1,
                    out=buffer_tensor_teacher_L1[n_cls_tokens : n_cls_tokens + n_masked_patches_L1],
                )
                tokens_after_head_L1 = self.teacher.dino_head(buffer_tensor_teacher_L1)
                teacher_cls_tokens_after_head_L1 = tokens_after_head_L1[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head_L1 = tokens_after_head_L1[
                    n_cls_tokens : n_cls_tokens + n_masked_patches_L1
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list_L0 = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head_L0, teacher_temp=teacher_temp
                ).view(n_L0_crops_teacher, -1, *teacher_cls_tokens_after_head_L0.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head_L0)
                teacher_dino_softmaxed_centered_list_L1 = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head_L1, teacher_temp=teacher_temp
                ).view(n_L1_crops_teacher, -1, *teacher_cls_tokens_after_head_L1.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head_L1)
               
                if do_ibot:
                    masked_teacher_patch_tokens_after_head_L0 = masked_teacher_patch_tokens_after_head_L0.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered_L0 = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head_L0[:, :n_masked_patches_L0], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered_L0 = masked_teacher_ibot_softmaxed_centered_L0.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head_L0[:n_masked_patches_L0])
                    masked_teacher_patch_tokens_after_head_L1 = masked_teacher_patch_tokens_after_head_L1.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered_L1 = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head_L1[:, :n_masked_patches_L1], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered_L1 = masked_teacher_ibot_softmaxed_centered_L1.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head_L1[:n_masked_patches_L1])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            if do_hmsa:
                return teacher_dino_softmaxed_centered_list_L0, masked_teacher_ibot_softmaxed_centered_L0,teacher_dino_softmaxed_centered_list_L1, masked_teacher_ibot_softmaxed_centered_L1, teacher_cls_tokens_after_head_L0,teacher_cls_tokens_after_head_L1,teacher_hmsa, ibot_teacher_patch_tokens_L0, ibot_teacher_patch_tokens_L1
            else:
                return teacher_dino_softmaxed_centered_list_L0, masked_teacher_ibot_softmaxed_centered_L0,teacher_dino_softmaxed_centered_list_L1, masked_teacher_ibot_softmaxed_centered_L1, teacher_cls_tokens_after_head_L0,teacher_cls_tokens_after_head_L1

        if do_hmsa:
            teacher_dino_softmaxed_centered_list_L0, masked_teacher_ibot_softmaxed_centered_L0,teacher_dino_softmaxed_centered_list_L1, masked_teacher_ibot_softmaxed_centered_L1, teacher_cls_tokens_after_head_L0,teacher_cls_tokens_after_head_L1, teacher_hmsa,ibot_teacher_patch_tokens_L0, ibot_teacher_patch_tokens_L1 = get_teacher_output()
        else:
            teacher_dino_softmaxed_centered_list_L0, masked_teacher_ibot_softmaxed_centered_L0,teacher_dino_softmaxed_centered_list_L1, masked_teacher_ibot_softmaxed_centered_L1, teacher_cls_tokens_after_head_L0,teacher_cls_tokens_after_head_L1 = get_teacher_output()
        
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop

        if do_hmsa:
            # student_output, student_hmsa = self.student.backbone(
            #     [L0_crops, L1_crops, L0_local_crops, L1_local_crops], masks=[masks_L0, masks_L1, None, None], L2_crops= L2_crops, do_hmsa=do_hmsa, is_training=True)
            student_output, student_hmsa = self.student.backbone(
                [L0_crops, L1_crops, L0_local_crops, L1_local_crops], masks=[masks_L0, masks_L1, None, None],  do_hmsa=do_hmsa, is_training=True)
            student_L0_backbone_output_dict, student_L1_backbone_output_dict, student_local_backbone_output_dict_L0, student_local_backbone_output_dict_L1 = student_output[0],student_output[1],student_output[2],student_output[3]
        else:
            student_L0_backbone_output_dict, student_L1_backbone_output_dict, student_local_backbone_output_dict_L0, student_local_backbone_output_dict_L1 = self.student.backbone(
                [L0_crops, L1_crops, L0_local_crops, L1_local_crops], masks=[masks_L0, masks_L1, None, None], do_hmsa=do_hmsa, is_training=True
            )

        inputs_for_student_head_list_L0, inputs_for_student_head_list_L1 = [], []


        # 1a: local crops cls tokens
        student_local_cls_tokens_L0 = student_local_backbone_output_dict_L0["x_norm_clstoken"]
        inputs_for_student_head_list_L0.append(student_local_cls_tokens_L0.unsqueeze(0))
        
        student_local_cls_tokens_L1 = student_local_backbone_output_dict_L1["x_norm_clstoken"]
        inputs_for_student_head_list_L1.append(student_local_cls_tokens_L1.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens_L0 = student_L0_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list_L0.append(student_global_cls_tokens_L0.unsqueeze(0))

        student_global_cls_tokens_L1 = student_L1_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list_L1.append(student_global_cls_tokens_L1.unsqueeze(0))

       

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_L0_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens_L0 = student_L0_backbone_output_dict["x_norm_patchtokens"]
            ibot_student_patch_tokens_L1 = student_L1_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens_L0 = ibot_student_patch_tokens_L0.new_zeros(upperbound_L0, _dim)
            buffer_tensor_patch_tokens_L1 = ibot_student_patch_tokens_L1.new_zeros(upperbound_L1, _dim)
            buffer_tensor_patch_tokens_L0[:n_masked_patches_L0].copy_(
                torch.index_select(ibot_student_patch_tokens_L0.flatten(0, 1), dim=0, index=mask_indices_list_L0)
            )
            buffer_tensor_patch_tokens_L1[:n_masked_patches_L1].copy_(
                torch.index_select(ibot_student_patch_tokens_L1.flatten(0, 1), dim=0, index=mask_indices_list_L1)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list_L0.append(buffer_tensor_patch_tokens_L0.unsqueeze(0))
                inputs_for_student_head_list_L1.append(buffer_tensor_patch_tokens_L1.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head_L0 = self.student.ibot_head(buffer_tensor_patch_tokens_L0)[
                    :n_masked_patches_L0
                ]
                student_global_masked_patch_tokens_after_head_L1 = self.student.ibot_head(buffer_tensor_patch_tokens_L1)[
                    :n_masked_patches_L1
                ]
            #print (ibot_student_patch_tokens_L0.shape,buffer_tensor_patch_tokens_L0.shape )
        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list_L0)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))
        #print (len(outputs_list))
        _attn_bias2, cat_inputs2 = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list_L1)
        outputs_list2 = _attn_bias2.split(self.student.dino_head(cat_inputs2))
        # print ('outputs_list2: ',len(outputs_list2))
        # print ('outputs_list: ',len(outputs_list))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head_L0 = outputs_list.pop(0).squeeze(0)
        student_local_cls_tokens_after_head_L1 = outputs_list2.pop(0).squeeze(0)
        # print ('student_local_cls_tokens_after_head', student_local_cls_tokens_after_head_L0.shape, student_local_cls_tokens_after_head_L1.shape)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head_L0 = outputs_list.pop(0).squeeze(0)
        student_global_cls_tokens_after_head_L1 = outputs_list2.pop(0).squeeze(0)
        # print ('student_global_cls_tokens_after_head', student_global_cls_tokens_after_head_L0.shape, student_global_cls_tokens_after_head_L1.shape)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head_L0 = outputs_list.pop(0).squeeze(0)[:n_masked_patches_L0]
            student_global_masked_patch_tokens_after_head_L1 = outputs_list2.pop(0).squeeze(0)[:n_masked_patches_L1]

        #print (len(student_local_cls_tokens_after_head_L0.chunk(n_local_crops)), teacher_dino_softmaxed_centered_list_L0.shape)
        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head_L0.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list_L0,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            dino_local_crops_loss_L1 = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head_L1.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list_L1,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            dino_local_crops_loss_L0L1 = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head_L0.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list_L1,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            
            dino_local_crops_loss_L1L0 = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head_L1.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list_L0,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = (dino_local_crops_loss + dino_local_crops_loss_L1 + dino_local_crops_loss_L0L1 + dino_local_crops_loss_L1L0 )/4

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * (dino_local_crops_loss + dino_local_crops_loss_L1 + dino_local_crops_loss_L0L1 + dino_local_crops_loss_L1L0 )

        # process global crops
        loss_scales = 1  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head_L0],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list_L0.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )
            
            dino_global_crops_loss2 = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head_L1],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list_L1.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            dino_global_crops_loss3 = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head_L0],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list_L1.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )
            #print (student_global_cls_tokens_after_head_L1.shape,teacher_dino_softmaxed_centered_list_L0.shape, teacher_dino_softmaxed_centered_list_L0.flatten(0, 1).shape )
            dino_global_crops_loss4 = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head_L1],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list_L0.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            #print ('teacher: ', teacher_dino_softmaxed_centered_list_L1.shape, teacher_dino_softmaxed_centered_list_L0.shape, teacher_dino_softmaxed_centered_list_L1.flatten(0, 1).shape)
            # dino_global_crops_loss_teacher = (
            #     self.dino_loss(
            #         student_output_list=[teacher_dino_softmaxed_centered_list_L0.float()],
            #         teacher_out_softmaxed_centered_list=[
            #             teacher_dino_softmaxed_centered_list_L1.flatten(0, 1).float()
            #         ],  # these were chunked and stacked in reverse so A is matched to B
            #     )
            #     * loss_scales
            #     / (n_global_crops_loss_terms + + n_local_crops_loss_terms)
            # )
            #print ('student: ', student_global_cls_tokens_after_head_L1.shape,student_global_cls_tokens_after_head_L0.unsqueeze(0).shape ,student_global_cls_tokens_after_head_L0.shape )
            # student_dino_softmaxed_centered_list_L0 = self.dino_loss.softmax_center_teacher(
            #         student_global_cls_tokens_after_head_L0, teacher_temp=teacher_temp
            #     ).view(1, -1, *student_global_cls_tokens_after_head_L0.shape[1:])
            # self.dino_loss.update_center(student_global_cls_tokens_after_head_L0)
            # student_dino_softmaxed_centered_list_L1 = self.dino_loss.softmax_center_teacher(
            #         student_global_cls_tokens_after_head_L1, teacher_temp=teacher_temp
            #     ).view(1, -1, *student_global_cls_tokens_after_head_L1.shape[1:])
            # self.dino_loss.update_center(student_global_cls_tokens_after_head_L1)
            # dino_global_crops_loss_student = (
            #     self.dino_loss(
            #         student_output_list=[student_dino_softmaxed_centered_list_L0],
            #         teacher_out_softmaxed_centered_list=[
            #             student_dino_softmaxed_centered_list_L1.flatten(0, 1)
            #         ],  # these were chunked and stacked in reverse so A is matched to B
            #     )
            #     * loss_scales
            #     / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            # )
            # print (dino_global_crops_loss,dino_global_crops_loss2 ,dino_global_crops_loss3, dino_global_crops_loss4, dino_global_crops_loss_teacher, dino_global_crops_loss_student)
            # # print ('dino_global_crops_loss_teacher:', dino_global_crops_loss_teacher)
            # # print ('dino_global_crops_loss_student:', dino_global_crops_loss_student)
            
            # teacher_dino_softmaxed_centered_list_L1_log = torch.log(teacher_dino_softmaxed_centered_list_L1)
            # kl_loss_teacher = F.kl_div(teacher_dino_softmaxed_centered_list_L1_log, teacher_dino_softmaxed_centered_list_L0, reduction="batchmean") 
            
            # student_global_cls_tokens_after_head_L1_log = torch.log(student_dino_softmaxed_centered_list_L1.float())
            # kl_loss_student = F.kl_div(student_global_cls_tokens_after_head_L1_log, student_dino_softmaxed_centered_list_L0.float(), reduction="batchmean") 

            
            # teacher_cls_tokens_after_head_L1 = F.normalize(teacher_cls_tokens_after_head_L1.float(), p=2, dim=-1)
            # teacher_cls_tokens_after_head_L0= F.normalize(teacher_cls_tokens_after_head_L0.float(), p=2, dim=-1)

            # simloss_teacher = F.cosine_similarity(teacher_cls_tokens_after_head_L1, teacher_cls_tokens_after_head_L0, dim=-1)#.clamp(min=-0.99, max=0.99)        
            # ms_loss_teacher = (1 - simloss_teacher.mean())

            
            
            # student_global_cls_tokens_after_head_L1 = F.normalize(student_global_cls_tokens_after_head_L1.float(), p=2, dim=-1)
            # student_global_cls_tokens_after_head_L0= F.normalize(student_global_cls_tokens_after_head_L0.float(), p=2, dim=-1)

            # simloss_student = F.cosine_similarity(student_global_cls_tokens_after_head_L1,student_global_cls_tokens_after_head_L0, dim=-1)#.clamp(min=-0.99, max=0.99)        
            # ms_loss_student = (1 - simloss_student.mean())

            # print ('kl loss: ', ms_loss_teacher, ms_loss_student)
           
            # loss_dict["ms_loss"] = (ms_loss_teacher + ms_loss_student)/2# + dino_global_crops_loss_teacher +dino_global_crops_loss_student)/6
            # loss_accumulator += self.dino_loss_weight * (ms_loss_teacher + ms_loss_student)
            
            loss_dict["dino_global_crops_loss"] = (dino_global_crops_loss + dino_global_crops_loss2 + dino_global_crops_loss3 + dino_global_crops_loss4)/4# + dino_global_crops_loss_teacher +dino_global_crops_loss_student)/6

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * (dino_global_crops_loss + dino_global_crops_loss2 + dino_global_crops_loss3 + dino_global_crops_loss) #+ dino_global_crops_loss_teacher +dino_global_crops_loss_student )
            
            
            student_cls_tokens_L0 = student_global_cls_tokens_L0
            student_cls_tokens_L1 = student_global_cls_tokens_L1

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens_L0.chunk(1)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens_L1.chunk(1)
                )
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if do_ibot:
            # compute loss
            #print ('ibot', student_global_masked_patch_tokens_after_head_L0.shape, masked_teacher_ibot_softmaxed_centered_L0.shape)
            ibot_patch_loss_L0 = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head_L0,
                    masked_teacher_ibot_softmaxed_centered_L0,
                    student_masks_flat=masks_L0,
                    n_masked_patches=n_masked_patches_L0,
                    masks_weight=masks_weight_L0,
                )
                * loss_scales
                * ibot_loss_scale
            )

            

            #print ('ibot L1', student_global_masked_patch_tokens_after_head_L1.shape, masked_teacher_ibot_softmaxed_centered_L1.shape)
            ibot_patch_loss_L1 = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head_L1,
                    masked_teacher_ibot_softmaxed_centered_L1,
                    student_masks_flat=masks_L1,
                    n_masked_patches=n_masked_patches_L1,
                    masks_weight=masks_weight_L1,
                )
                * loss_scales
                * ibot_loss_scale
            )
            # if student_global_masked_patch_tokens_after_head_L1.shape[0] == student_global_masked_patch_tokens_after_head_L0.shape[0]:
            #     #print ('ibot L1L0', student_global_masked_patch_tokens_after_head_L1.shape, masked_teacher_ibot_softmaxed_centered_L0.shape)
            #     ibot_patch_loss_L1L0 = (
            #         self.ibot_patch_loss.forward_masked(
            #             student_global_masked_patch_tokens_after_head_L1,
            #             student_global_masked_patch_tokens_after_head_L0,
            #             student_masks_flat=masks_L1,
            #             n_masked_patches=n_masked_patches_L1,
            #             masks_weight=masks_weight_L1,
            #         )
            #         * loss_scales
            #         * ibot_loss_scale
            #     )

            #     #print ('ibot L0L1', student_global_masked_patch_tokens_after_head_L0.shape, masked_teacher_ibot_softmaxed_centered_L1.shape)
            #     ibot_patch_loss_L0L1 = (
            #         self.ibot_patch_loss.forward_masked(
            #             student_global_masked_patch_tokens_after_head_L0,
            #             student_global_masked_patch_tokens_after_head_L1,
            #             student_masks_flat=masks_L0,
            #             n_masked_patches=n_masked_patches_L0,
            #             masks_weight=masks_weight_L0,
            #         )
            #         * loss_scales
            #         * ibot_loss_scale
            #     )
            #     loss_dict["ibot_loss"] = (ibot_patch_loss_L0 + ibot_patch_loss_L1 + ibot_patch_loss_L1L0 + ibot_patch_loss_L0L1) / 4

            #                 # accumulate loss
            #     loss_accumulator += self.ibot_loss_weight * (ibot_patch_loss_L0 + ibot_patch_loss_L1 + ibot_patch_loss_L1L0 + ibot_patch_loss_L0L1)
            #else:
            # store for display
            loss_dict["ibot_loss"] = (ibot_patch_loss_L0 + ibot_patch_loss_L1 ) / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * (ibot_patch_loss_L0 + ibot_patch_loss_L1)

        
        if do_kd:
            fm_crops = images["collated_fm_crops"].cuda(non_blocking=True)
            #fm_crops2 = images["collated_fm_crops2"].cuda(non_blocking=True)
            fm_crops = torch.nan_to_num(fm_crops, nan=0.0)
            #print ('fm_crops: ',fm_crops.shape)
            if torch.isnan(fm_crops).any():
                print("NaN detected in fm_crops!")
            
            #bat = len(ibot_student_patch_tokens)//2
            dim = fm_crops.shape[-1]
            student_global_patch_tokens = ibot_student_patch_tokens_L0 #student_global_backbone_output_dict['x_norm_patchtokens']
            #print ('student_global_patch_tokens',student_global_patch_tokens.shape)
            #print ("KD ")
            fm_crops_ = fm_crops#self.mlp(fm_crops.to(torch.float32))#
            t_ptoken_ = fm_crops_#.view(-1, 1024)#torch.cat([fm_crops_.float(), fm_crops_.float()])
            s_ptoken = student_global_patch_tokens.float()
            
            t_ptoken_ = F.normalize(t_ptoken_.float(), p=2, dim=-1)
            s_ptoken = F.normalize(s_ptoken.float(), p=2, dim=-1)

            simloss = F.cosine_similarity(t_ptoken_.float(), s_ptoken.float(), dim=-1)#.clamp(min=-0.99, max=0.99)        
            
            # fm_crops2_ = self.mlp2(fm_crops2.to(torch.float32))#
            # t_ptoken2_ = fm_crops2_#.view(-1, 1024)#torch.cat([fm_crops_.float(), fm_crops_.float()])
            
            
            # t_ptoken2_ = F.normalize(t_ptoken2_.float(), p=2, dim=-1)

            # simloss2 = F.cosine_similarity(t_ptoken2_.float(), s_ptoken.float(), dim=-1)#.clamp(min=-0.99, max=0.99)        
            
            kd_loss = (1 - simloss.mean())# + (1 - simloss2.mean()) 


            loss_dict["kd_loss"] = kd_loss 
            
            #print ('loss accumulator: ', loss_accumulator)
            loss_accumulator += self.kd_loss_weight *   kd_loss
        
        if do_hmsa:
            
            student_features = F.normalize(student_hmsa, p=2, dim=-1)
            teacher_features = F.normalize(teacher_hmsa , p=2, dim=-1)  
            cosine_sim = F.cosine_similarity(teacher_features.float(),student_features.float(), dim=-1)
            temperature = 0.1 
            cosine_loss =  (1 - cosine_sim.mean())/ temperature


            hmsa_loss = cosine_loss 
            #print ('hmsa_loss', hmsa_loss.float())
            loss_dict["hmsa_loss"] = hmsa_loss 
            loss_accumulator += self.hmsa_loss_weight * hmsa_loss

            # teacher_patch_tokens_L0 = F.normalize(ibot_teacher_patch_tokens_L0, p=2, dim=-1)
            # teacher_patch_tokens_L1 = F.normalize(ibot_teacher_patch_tokens_L1, p=2, dim=-1)
            # student_patch_tokens_L0 = F.normalize(ibot_student_patch_tokens_L0, p=2, dim=-1) 
            # student_patch_tokens_L1 = F.normalize(ibot_student_patch_tokens_L1, p=2, dim=-1)
            # #print (ibot_teacher_patch_tokens_L0.shape,ibot_teacher_patch_tokens_L1.shape, ibot_student_patch_tokens_L0.shape,ibot_student_patch_tokens_L1.shape  )

            # def cosine_similarity_loss_no_mean(L0, L1, reduction='mean'):
            #     b, _, d = L0.shape  # Get batch size and feature dimension11
            #     # Reshape L0 into [b, 16, 16, 1024] to group each set of 16 tokens
            #     L0_grouped = L0.view(b, 16, 16, d)  # [b, 16, 16, 1024]
            #     L1 = L1[:,:16,:]
                
            #     L1_grouped = L1.view(b, 16, 1, d) 
            #     #print (L0_grouped.shape, L1_grouped.shape)

            #     # Expand L1 to match L0's shape for pairwise similarity: [b, 16, 1, 1024]
            #     #L1_expanded = L1.unsqueeze(2)  # [b, 16, 1, 1024]
            #     # Compute cosine similarity for each pair in the 16-token group
            #     #print (L0_grouped.shape, L1_expanded.shape)

            #     # Reshape for cosine similarity
            #     L0_flat = L0_grouped.view(b, 16, 16, d)  # Shape: [b, 16, 16, 1, d]
            #     L1_flat = L1_grouped.view(b, 16, 1, d)   # Shape: [b, 16, 1, 2, d]
            #     cos_sim = F.cosine_similarity(L0_flat, L1_flat, dim=-1) # Shape: [b, 16, 16]
            #     # Aggregate similarity values across the 16 patches in L0
            #     if reduction == 'mean':
            #         cos_sim_agg = cos_sim.mean(dim=-1)  # [b, 16]
            #     elif reduction == 'max':
            #         cos_sim_agg = cos_sim.max(dim=-1).values  # [b, 16]
            #     elif reduction == 'min':
            #         cos_sim_agg = cos_sim.min(dim=-1).values  # [b, 16]
            #     else:
            #         raise ValueError("reduction must be 'mean', 'max', or 'min'")
            #     # Convert to loss (maximize similarity → minimize negative cosine similarity)
            #     loss = 1 - cos_sim_agg.mean()
            #     return loss

            # tL0_sL1 = cosine_similarity_loss_no_mean(teacher_patch_tokens_L0, student_patch_tokens_L1)
            # tL1_sL0 = cosine_similarity_loss_no_mean(student_patch_tokens_L0 , teacher_patch_tokens_L1)
            # print ('cross scale:', tL0_sL1.item(), tL1_sL0.item())
            # loss_dict["cross_scale"] = (tL0_sL1 + tL1_sL0)/2
            # loss_accumulator += self.hmsa_loss_weight *  (tL0_sL1 + tL1_sL0)


            
        
        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.dino_head._streams = (
                self.teacher.dino_head._streams
            ) = self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])