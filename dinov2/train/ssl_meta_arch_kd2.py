# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn
#torch.autograd.set_detect_anomaly(True)

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg_kd
from dinov2.layers import DINOHead, HMSA_head
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model
import torch.nn.functional as F
from dinov2.models.vision_transformer_kd import BlockChunk


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg_kd(cfg)
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
        self.do_kd = cfg.kd.loss_weight > 0
        self.do_hmsa = cfg.hmsa.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

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

        if self.do_kd:
            self.kd_loss_weight = cfg.kd.loss_weight

        if self.do_hmsa:
            self.hmsa_loss_weight = cfg.hmsa.loss_weight
        

           

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

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

        do_dino = self.do_dino
        do_ibot = self.do_ibot
        do_kd = self.do_kd
        do_hmsa = self.do_hmsa

        n_local_crops = self.cfg.crops.local_crops_number

        if  n_local_crops > 0:
            n_global_crops = 2
            assert n_global_crops == 2
        else: 
            n_global_crops = 1

        
        
        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        if  n_local_crops > 0:
            local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        if do_hmsa:
            L1_crops = images["collated_L1_crops"].cuda(non_blocking=True)
            L2_crops = images["collated_L2_crops"].cuda(non_blocking=True)
        
        #print ('fm_crops: ',fm_crops.shape)
        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        if  n_local_crops > 0:
            n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
            n_globateacher_cls_tokensl_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            
            if do_hmsa:
                teacher_backbone_output_dict = self.teacher.backbone(x, L1 = L1_crops, is_training=True)
            else:
                teacher_backbone_output_dict = self.teacher.backbone(x , is_training=True)
                #print (len(teacher_backbone_output_dict))
            
            #print(teacher_backbone_output_dict)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            #print (teacher_cls_tokens.shape, teacher_patch_tokens.shape)
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            if n_global_crops > 1:
                teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
                n_cls_tokens = teacher_cls_tokens.shape[0]
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            
            #print ('n_cls_tokens: ', n_cls_tokens)
            if  n_local_crops > 0:
                if do_ibot and not self.ibot_separate_head:
                    #print ('do_ibot and not self.ibot_separate_head')
                    buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                    buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                    torch.index_select(
                        ibot_teacher_patch_tokens.flatten(0, 1),
                        dim=0,
                        index=mask_indices_list,
                        out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                    )
                    tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                    teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                    masked_teacher_patch_tokens_after_head = tokens_after_head[
                        n_cls_tokens : n_cls_tokens + n_masked_patches
                    ]
                    if do_hmsa:
                        teacher_hmsa = teacher_backbone_output_dict["hmsa_feat"]
                        tokens_after_head_teacher_hmsa = self.teacher.dino_head(teacher_hmsa)
        
                elif do_ibot and self.ibot_separate_head:
                    #print ('do_ibot and self.ibot_separate_head')
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
                   #print ('teacher w/o ibot head')
                   teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                   masked_teacher_ibot_softmaxed_centered = None

                if self.cfg.train.centering == "centering":
                    teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                        teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                    ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                    self.dino_loss.update_center(teacher_cls_tokens_after_head)
                    if do_ibot:
                        masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                        masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                            masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                        )
                        masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                        self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

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
                return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered, teacher_patch_tokens, teacher_backbone_output_dict, tokens_after_head_teacher_hmsa
            elif n_local_crops == 0:
                return teacher_cls_tokens, teacher_patch_tokens
            else:
                return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered, teacher_patch_tokens, teacher_backbone_output_dict 

        if do_hmsa:
            teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered, teacher_patch_tokens, teacher_backbone_output_dict, tokens_after_head_teacher_hmsa = get_teacher_output()
        elif n_local_crops == 0:
            teacher_cls_tokens, teacher_patch_tokens = get_teacher_output()
        else:
            teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered, teacher_patch_tokens, teacher_backbone_output_dict = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        
        if do_hmsa:
            student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
                [global_crops, local_crops],masks=[masks, None], L1 = [L1_crops, None], is_training=True)# masks=[masks, None], is_training=True)
        elif n_local_crops == 0:
            student_global_backbone_output_dict = self.student.backbone(
                global_crops,  is_training=True)# masks=[masks, None], is_training=True)
        else:
            student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
                [global_crops, local_crops],masks=[masks, None],  is_training=True)# masks=[masks, None], is_training=True)

            
        inputs_for_student_head_list = []


        
        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
       
                
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]
                #print ('student_global_masked_patch_tokens_after_head 1 ', student_global_masked_patch_tokens_after_head.shape)
        
        if do_hmsa: 
    
            student_hmsa = student_global_backbone_output_dict["hmsa_feat"]
            tokens_after_head_student_hmsa = self.student.dino_head(student_hmsa)

        if  n_local_crops > 0:
            # 2: run
            _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
            outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

            # 3a: local crops cls tokens
            student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
            #print (torch.stack(student_local_cls_tokens_after_head.shape))

            # 3b: global crops cls tokens
            student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

            # 3c: global crops patch tokens
            if do_ibot and not self.ibot_separate_head:
                student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]
            # print ('student_global_masked_patch_tokens_after_hea 3',student_global_masked_patch_tokens_after_head.shape)
            
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            #print (dino_local_crops_loss)
            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

            # process global crops
            loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
            #print ('dinoloss: ', dino_global_crops_loss)
            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually
                #print ('koleo_loss: ', koleo_loss)

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2
            #print ('ibot_loss', ibot_patch_loss / 2)

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        if do_kd:
            fm_crops = images["collated_fm_crops"].cuda(non_blocking=True)
            fm_crops = torch.nan_to_num(fm_crops, nan=0.0)
            #print ('fm_crops: ',fm_crops.shape)
            if torch.isnan(fm_crops).any():
                print("NaN detected in fm_crops!")

            
            
            #bat = len(ibot_student_patch_tokens)//2
            dim = fm_crops.shape[-1]
            student_global_patch_tokens = student_global_backbone_output_dict['x_norm_patchtokens']
            #print ('student_global_patch_tokens:',student_global_patch_tokens.shape)
            #fm_crops_ = fm_crops.view(-1, 256, dim)
            
            fm_crops_ = fm_crops
            t_ptoken_ = fm_crops_.float()
            s_ptoken = student_global_patch_tokens.float()
            
            t_ptoken_ = F.normalize(t_ptoken_.float(), p=2, dim=-1)
            s_ptoken = F.normalize(s_ptoken.float(), p=2, dim=-1)

            simloss = F.cosine_similarity(t_ptoken_.float(), s_ptoken.float(), dim=-1)#.clamp(min=-0.99, max=0.99)        
            kd_loss = (1 - simloss.mean())
            
            #t_ptoken_ = t_ptoken_ / torch.clamp(t_ptoken_.sum(dim=-1, keepdim=True), min=1e-6)
            #s_ptoken = s_ptoken / torch.clamp(s_ptoken.sum(dim=-1, keepdim=True), min=1e-6)

            # Clamp values to avoid log(0)
            #s_ptoken = torch.clamp(s_ptoken, min=1e-6, max=1-1e-6)
            #t_ptoken_ = torch.clamp(t_ptoken_, min=1e-6, max=1-1e-6)
            # s_ptoken = (s_ptoken - s_ptoken.mean(dim=-1, keepdim=True)) / (s_ptoken.std(dim=-1, keepdim=True) + 1e-6)
            # t_ptoken_ = (t_ptoken_ - t_ptoken_.mean(dim=-1, keepdim=True)) / (t_ptoken_.std(dim=-1, keepdim=True) + 1e-6)

            # # Compute log softmax safely
            # temperature = 5  # Example temperature# Apply softmax to teacher's logits to get probabilities
            # t_ptoken_ = F.softmax(t_ptoken_/temperature, dim=-1)

            # # Apply softmax to the student's logits and then log them
            # s_ptoken = F.log_softmax(s_ptoken, dim=-1)

            # # Compute KL divergence loss
            # kd_loss = F.kl_div(s_ptoken, t_ptoken_, reduction="batchmean")
     

            #print ('kd_loss:', kd_loss)

            def sinkhorn_knopp(scores, n_iters=5, epsilon=0.05):
                """
                Sinkhorn-Knopp Normalization to obtain a doubly stochastic matrix.
                A
                    scores: (B, N, N) - Pairwise similarity matrix
                    n_iters: Number of iterations
                    epsilon: Small value for numerical stability
                Returns:
                    Normalized doubly stochastic matrix
                """
                B, N, _ = scores.shape  # Unpack the batch, token number, and dimensions (tokens x tokens)
                scores = scores.float()  # Ensure the matrix is in float32 for stability
                scores = scores / epsilon  # Scale for stability

                for _ in range(n_iters):
                    # Row normalization (log-softmax)
                    scores = scores - scores.logsumexp(dim=1, keepdim=True)
                    # Column normalization (log-softmax)
                    scores = scores - scores.logsumexp(dim=2, keepdim=True)

            #     return scores.exp()  # Convert back to probabilities
                
            # # Assuming t_ptoken_ and s_ptoken have shape [B, token_number, dim]
            # B, N, D = t_ptoken_.shape  # Get B=batch_size, N=number of tokens, D=dimensionality

            # # Compute the pairwise similarity between the teacher and student features
            # cost_matrix = F.cosine_similarity(t_ptoken_.unsqueeze(2), s_ptoken.unsqueeze(1), dim=-1)  # Shape: [B, N, N]

            # # Apply Sinkhorn-Knopp to normalize the cost matrix
            # sinkhorn_dist = sinkhorn_knopp(cost_matrix)

            # # Compute the softmax over the teacher tokens for comparison
            # teacher_softmax = F.softmax(t_ptoken_, dim=-1)  # Shape: [B, N, D]

            # # Now, compute MSE loss between the distributions (for the same shape)
            # # We need to ensure that the tensors have the same shape: [B, N, N]
            # sinkhorn_dist_expanded = sinkhorn_dist.unsqueeze(-1)  # Expand to [B, N, N, 1] for broadcasting
            # teacher_softmax_expanded = teacher_softmax.unsqueeze(2)  # Expand to [B, N, 1, D]

            # # Calculate the MSE loss for the distributions (adjust the dimensions as needed)
            # kd_loss_ = F.mse_loss(sinkhorn_dist_expanded, teacher_softmax_expanded)
            # #kd_loss = kd_loss.view(1)
            # kd_loss = self.fp16_scaler.scale(kd_loss_.clone().detach().to(device=kd_loss_.device)).item()


            loss_dict["kd_loss"] = kd_loss 
            
            #print ('loss accumulator: ', loss_accumulator)
            loss_accumulator += self.kd_loss_weight *   kd_loss


        def mse_loss(student_features, teacher_features):
            return F.mse_loss(student_features, teacher_features)
        
        def contrastive_loss(student_features, teacher_features, temperature=0.07): ##infoNCE
            student_features = F.normalize(student_features, dim=-1)
            teacher_features = F.normalize(teacher_features, dim=-1)

            batch_size = student_features.shape[0]
            similarity_matrix = torch.mm(student_features, teacher_features.T)  # Cosine similarity

            labels = torch.arange(batch_size).to(student_features.device)  # Positive pairs
            loss = F.cross_entropy(similarity_matrix / temperature, labels)
            return loss


        if do_hmsa:
            teacher_hmsa = teacher_backbone_output_dict["hmsa_feat"]
            student_hmsa = student_global_backbone_output_dict["hmsa_feat"]
            batch_size, token_size, feature_dim = teacher_hmsa.shape
            student_features = student_hmsa.reshape(batch_size * token_size, feature_dim) 
            teacher_features = teacher_hmsa.reshape(batch_size * token_size, feature_dim)
            #student_features = student_features + torch.randn_like(student_features) * 0.01
            #teacher_features = teacher_features + torch.randn_like(teacher_features) * 0.01

            student_features = F.normalize(student_features, p=2, dim=-1)
            teacher_features = F.normalize(teacher_features , p=2, dim=-1)   
            simloss_hmsa = F.cosine_similarity(teacher_features.float(),student_features .float(), dim=-1)
            hmsa_loss = (1 - simloss_hmsa.mean())
            #print ('teacher_hmsa:' , teacher_hmsa)
            #print ('student_hmsa: ', student_hmsa)
            #hmsa_loss = mse_loss(student_features, teacher_features)
            #print ('hmsa_loss', hmsa_loss.float())
            
            loss_dict["hmsa_loss"] = hmsa_loss 
            loss_accumulator += self.hmsa_loss_weight * hmsa_loss
            
            
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