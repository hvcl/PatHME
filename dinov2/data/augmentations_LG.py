import logging, torch, random

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")




class DataAugmentationDINO_LG(object):
    def __init__(self,  local_crops_number, kd, hmsa):
        # Define augmentation transformations for feature vectors
        
        self.local_crops_number = local_crops_number
        self.kd = kd
        self.hmsa = hmsa
    
    def random_masking(self, feature, mask_ratio):
        """
        Apply random masking to the feature vectors.

        Args:
            feature: Input feature tensor of shape (batch_size, feature_dim).
            mask_ratio: Fraction of features to mask.

        Returns:
            Masked feature tensor.
        """
        batch_size, feature_dim = feature.shape
        # Generate a random mask
        mask = (torch.rand(batch_size, feature_dim) < mask_ratio).float()
        # Apply the mask (element-wise multiplication)
        masked_feature = feature * mask
        return masked_feature

    
    def random_select(self, feature, num_samples=8):
        """
        Randomly select `num_samples` items from the input feature.

        Args:
            feature: Input feature tensor of shape (batch_size, feature_dim).
            num_samples: Number of samples to select.

        Returns:
            Selected feature tensor.
        """
        batch_size = feature.shape[0]
        indices = torch.randperm(batch_size)[:num_samples]  # Select random indices
        return feature[indices]

    def random_crop(self, feature, fm_feature = None, crop_size=196):
            """
            Randomly crops a feature tensor from shape [256, feature_dim] to [224, feature_dim].
            
            Args:
                feature (torch.Tensor): Input feature tensor of shape (256, feature_dim).
                crop_size (int): Target crop size along the first dimension.
            
            Returns:
                torch.Tensor: Cropped feature tensor of shape (224, feature_dim).
            """
            assert feature.shape[0] >= crop_size, "Input must have at least {} rows".format(crop_size)
            
            start_idx = torch.randint(0, feature.shape[0] - crop_size + 1, (1,)).item()  # Random start index
            cropped_feature = feature[start_idx : start_idx + crop_size, :]
            if fm_feature != None:
                fm_cropped_feature = fm_feature[start_idx : start_idx + crop_size, :]
                return cropped_feature, fm_cropped_feature
            else:
                return cropped_feature

    def __call__(self, features):
        
        output = {}
        if self.kd > 0:
            fm_feature = torch.tensor(features[1][-256:,0,:])#
            fm_feature = torch.nan_to_num(fm_feature, nan=0.0)
            #print ('fm feature: ', fm_feature.shape)
            feature = torch.tensor(features[0])
            feature = torch.nan_to_num(feature, nan=0.0)
            L0_feat = feature[-256:, :]#.clone().detach()
            #print(L0_feat.shape, fm_feature.shape)
            #global_crop_1, fm_feat1 = self.random_crop(L0_feat, fm_feature)
            #global_crop_2, fm_feat2 = self.random_crop(L0_feat, fm_feature)
            output["fm_features"] = [fm_feature, fm_feature]
        else:
            feature = torch.tensor(features)
            L0_feat = feature[-256:, :]
            global_crop_1 = L0_feat#self.random_crop(L0_feat)
            global_crop_2 = L0_feat#self.random_crop(L0_feat)
            reshape_L0 = 16
            #feature = torch.nan_to_num(feature, nan=0.0)
        #feature = features.clone().detach() 
        if self.hmsa > 0:
            L2_feat = feature[:1,:]
            L1_feat = feature[1:17,:]
            output["L1_feat"] = L1_feat.unsqueeze(0)
            output["L2_feat"] = L2_feat.unsqueeze(0)
        if self.local_crops_number == 1:
            L1_feat = feature[1:17,:]
        L1_feat = feature[1:17,:]
        #local_crop_1 = L1_feat#self.random_crop(L0_feat)
        local_crop_2 = L1_feat
        reshape_L1 = 4
        
        output["L0_crops"]  = global_crop_1.transpose(1,0).view(-1,reshape_L0,reshape_L0).unsqueeze(0)
        output["L1_crops"] = local_crop_2.transpose(1,0).view(-1,reshape_L1,reshape_L1).unsqueeze(0)
        output["L0_crops_teacher"] = global_crop_1.transpose(1,0).view(-1,reshape_L0,reshape_L0).unsqueeze(0)
        output["L0_crops_teacher"] = local_crop_2.transpose(1,0).view(-1,reshape_L1,reshape_L1).unsqueeze(0)
        
        #masked_L0_feat = self.random_masking(L0_feat, mask_ratio=0.3)
        if self.local_crops_number == 1:
            local_crops = L0_feat.transpose(1,0).view(-1,16,16).unsqueeze(0)
        else:
            local_crops1 = [
            self.random_select(L0_feat, num_samples=36).transpose(1,0).view(-1,6,6) for _ in range(self.local_crops_number)
            ]
            local_crops2 = [
            self.random_select(L1_feat, num_samples=9).transpose(1,0).view(-1,3,3) for _ in range(self.local_crops_number)
            ]
        output["L0_local_crops"] = local_crops1
        output["L1_local_crops"] = local_crops2
        output["offsets"] = ()
        
    
        return output


class DataAugmentationDINO_kd(object):
    def __init__(self,  local_crops_number, kd, hmsa):
        # Define augmentation transformations for feature vectors
        
        self.local_crops_number = local_crops_number
        self.kd = kd
        self.hmsa = hmsa
    
    def random_masking(self, feature, mask_ratio):
        """
        Apply random masking to the feature vectors.

        Args:
            feature: Input feature tensor of shape (batch_size, feature_dim).
            mask_ratio: Fraction of features to mask.

        Returns:
            Masked feature tensor.
        """
        batch_size, feature_dim = feature.shape
        # Generate a random mask
        mask = (torch.rand(batch_size, feature_dim) < mask_ratio).float()
        # Apply the mask (element-wise multiplication)
        masked_feature = feature * mask
        return masked_feature

    
    def random_select(self, feature, num_samples=8):
        """
        Randomly select `num_samples` items from the input feature.

        Args:
            feature: Input feature tensor of shape (batch_size, feature_dim).
            num_samples: Number of samples to select.

        Returns:
            Selected feature tensor.
        """
        batch_size = feature.shape[0]
        indices = torch.randperm(batch_size)[:num_samples]  # Select random indices
        return feature[indices]

    def random_crop(self, feature, fm_feature = None, crop_size=196):
            """
            Randomly crops a feature tensor from shape [256, feature_dim] to [224, feature_dim].
            
            Args:
                feature (torch.Tensor): Input feature tensor of shape (256, feature_dim).
                crop_size (int): Target crop size along the first dimension.
            
            Returns:
                torch.Tensor: Cropped feature tensor of shape (224, feature_dim).
            """
            assert feature.shape[0] >= crop_size, "Input must have at least {} rows".format(crop_size)
            
            start_idx = torch.randint(0, feature.shape[0] - crop_size + 1, (1,)).item()  # Random start index
            cropped_feature = feature[start_idx : start_idx + crop_size, :]
            if fm_feature != None:
                fm_cropped_feature = fm_feature[start_idx : start_idx + crop_size, :]
                return cropped_feature, fm_cropped_feature
            else:
                return cropped_feature

    def __call__(self, features):
        
        output = {}
        if self.kd > 0:
            fm_feature = torch.tensor(features[1][-256:,0,:])#
            fm_feature = torch.nan_to_num(fm_feature, nan=0.0)
            #print ('fm feature: ', fm_feature.shape)
            feature = torch.tensor(features[0])
            feature = torch.nan_to_num(feature, nan=0.0)
            L0_feat = feature[-256:, :]#.clone().detach()
            #print(L0_feat.shape, fm_feature.shape)
            #global_crop_1, fm_feat1 = self.random_crop(L0_feat, fm_feature)
            #global_crop_2, fm_feat2 = self.random_crop(L0_feat, fm_feature)
            output["fm_features"] = [fm_feature, fm_feature]#[fm_feat1, fm_feat2]
            
        else:
            feature = torch.tensor(features)
            L0_feat = feature[-256:, :]
            global_crop_1 = self.random_crop(L0_feat)
            global_crop_2 = self.random_crop(L0_feat)
            reshape_ = 14
            #feature = torch.nan_to_num(feature, nan=0.0)
        #feature = features.clone().detach() 
        if self.hmsa > 0:
            L2_feat = feature[:1,:]
            L1_feat = feature[1:17,:]
            output["L1_feat"] = [L1_feat, L1_feat]
            output["L2_feat"] = L2_feat.unsqueeze(0)
        #if self.local_crops_number == 1:
        #    L1_feat = feature[1:17,:]
        #    global_crop_1 = L1_feat#self.random_crop(L0_feat)
        #    global_crop_2 = L1_feat
        #   reshape_= 4
        L0_feat = feature[-256:, :]
        global_crop_1 = L0_feat#self.random_crop(L0_feat)
        global_crop_2 = L0_feat#self.random_crop(L0_feat)
        reshape_ = 16
        output["global_crops"]  = [global_crop_1.transpose(1,0).reshape(-1,reshape_,reshape_), global_crop_2.transpose(1,0).reshape(-1,reshape_,reshape_)]
        output["global_crops_teacher"] = [global_crop_1.transpose(1,0).reshape(-1,reshape_,reshape_), global_crop_2.transpose(1,0).reshape(-1,reshape_,reshape_)]
        #masked_L0_feat = self.random_masking(L0_feat, mask_ratio=0.3)
        #if self.local_crops_number == 1:
        #local_crops = L0_feat.transpose(1,0).view(-1,16,16)
        #else:
        local_crops = [
        self.random_select(L0_feat, num_samples=36).transpose(1,0).view(-1,6,6) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()
        #print (output.keys())
        
    
        return output


     

class DataAugmentationDINO(object):
    def __init__(
        self,
        local_crops_number,
    ):
        self.local_crops_number = local_crops_number
    

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info("###################################")


    
    def local_transfo(self, x, selected_number):
        # Randomly sample a portion of the items in x
        return torch.stack(random.sample(list(x), selected_number))

    def __call__(self, image):
        output = {}
        #print (image.shape)
        
        if image.shape[0] != 3:
            image =  image.permute(1,0,2)

        #print ('image: ', image.shape ,image.dtype)
        global_crop_1 = image[0].to(torch.float16) 
        global_crop_2 = image[1].to(torch.float16)
        #print ('before: ' ,global_crop_1.shape, global_crop_2.shape)

        # max_items = 10000
        # def pad_or_trim(tensor, max_items):
        #     size = tensor.size(0)
        #     if size > max_items:
        #         return tensor[:max_items]
        #     elif size < max_items:
        #         padding = torch.zeros((max_items - size, tensor.size(1)), dtype=tensor.dtype, device=tensor.device)
        #         return torch.cat((tensor, padding), dim=0)
        #     else:
        #         return tensor
        
        # global_crop_1 = pad_or_trim(global_crop_1, max_items)
        # global_crop_2 = pad_or_trim(global_crop_2, max_items)

        #print ('after: ' ,global_crop_1.shape, global_crop_2.shape)


        output["global_crops"] = [global_crop_1.transpose(0,1), global_crop_2.transpose(0,1)]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1.transpose(0,1), global_crop_2.transpose(0,1)]

        # local crops:
        selected_number = int(len(image[1]) * (3/8))#int(max_items*(3/8))#
        local_crops =[]
        for _ in range(self.local_crops_number//2):
            local_augmented1 = self.local_transfo(global_crop_1, selected_number).transpose(0,1)
            local_augmented2 = self.local_transfo(global_crop_2, selected_number).transpose(0,1)
            local_crops.append(local_augmented1)
            local_crops.append(local_augmented2)

        #print ('local: ', torch.stack(local_crops).shape)
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output



class DataAugmentationDINO_s2(object):
    def __init__(
        self,
        local_crops_number, 
        patch_num,
    ):
        self.local_crops_number = local_crops_number
        self.patch_num = patch_num
    

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"local_crops_number: {local_crops_number}")

        
        logger.info("###################################")


    
    def local_transfo(self, x, selected_number):
        # Randomly sample a portion of the items in x
        if len(x) >= selected_number:

            return torch.stack(random.sample(list(x), selected_number))
        else:
            return torch.stack(random.choices(list(x), k=selected_number))

    def __call__(self, image):
        output = {}

        aug_1 = image[0] 
        aug_2 = image[2]

        #print (image.shape,'Aug shape: ', aug_1.shape, aug_2.shape)
        if len(aug_1) >= self.patch_num:
            global_crop_1 = aug_1[:self.patch_num] 
            global_crop_2 = aug_2[:self.patch_num]
        else:        
            num_repeats = self.patch_num // aug_1.size(0) + 1 
            gcrop_1 = aug_1.repeat((num_repeats, 1))
            gcrop_2 = aug_2.repeat((num_repeats,  1))
            global_crop_1 = gcrop_1[:self.patch_num]
            global_crop_2 = gcrop_2[:self.patch_num]
            

        output["global_crops"] = [global_crop_1.transpose(0,1), global_crop_2.transpose(0,1)]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1.transpose(0,1), global_crop_2.transpose(0,1)]

        # local crops:
        selected_number = int(self.patch_num * (3/8))
        local_crops =[]
        for _ in range(self.local_crops_number//2):
            local_augmented1 = self.local_transfo(aug_1, selected_number).transpose(0,1)
            local_augmented2 = self.local_transfo(aug_2, selected_number).transpose(0,1)
            #print (selected_number, aug_1.shape, 'local_augmented1', local_augmented1.shape, 'local_augmented2', local_augmented2.shape)
            local_crops.append(local_augmented1)
            local_crops.append(local_augmented2)

        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output



