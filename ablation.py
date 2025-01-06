import os
import glob
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import pickle
import nibabel as nib
from torchvision.models import vgg19
from torchvision.models import resnet50
from timm.models.vision_transformer import vit_base_patch16_224
import albumentations as A
from utils import seeding, create_dir, plot_metrics
from metrics import (DiceLoss, DiceBCELoss, DiceLossMultiClass, precision, recall, F2, dice_score, jac_score,
    precision_multiclass, recall_multiclass, F2_multiclass, dice_score_multiclass, jac_score_multiclass, hausdorff_distance, iou_score, iou_score_multiclass, hd95, normalized_surface_dice)
from sklearn.model_selection import train_test_split

# Load task config
with open('task_config.yaml', 'r') as f:
    task_configs = yaml.safe_load(f)

class MSDDataset(Dataset):
    def __init__(self, image_paths, mask_paths, modalities=None, slice_axis=2, transform=None):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.modalities = modalities  # Should be a list of integers
        self.slice_axis = slice_axis
        self.transform = transform  # Albumentations transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        img = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Handle modalities (channels)
        if img.ndim == 4:
            # Multiple modalities
            if self.modalities is not None:
                img = img[..., self.modalities]  # Select specified modalities using indices
            channels = img.shape[-1]
        else:
            # Single modality
            img = img[..., np.newaxis]
            channels = 1

        # Per-modality normalization
        for c in range(channels):
            modality = img[..., c]
            img[..., c] = (modality - modality.min()) / (modality.max() - modality.min() + 1e-8)

        # Extract a slice along the specified axis
        mid_slice = img.shape[self.slice_axis] // 2
        img = np.take(img, mid_slice, axis=self.slice_axis)
        mask = np.take(mask, mid_slice, axis=self.slice_axis)

        # Adjust mask labels (e.g., for binary segmentation)
        mask = np.where(mask > 0, 1, 0)
        mask = mask.astype(np.int64)

        # Handle dimensions after slicing
        if img.ndim == 2:
            # Shape: [H, W]
            img = img[np.newaxis, ...]  # Shape: [1, H, W]
        elif img.ndim == 3:
            if img.shape[-1] == 1:
                # Shape: [H, W, 1]
                img = img.squeeze(-1)  # Shape: [H, W]
                img = img[np.newaxis, ...]  # Shape: [1, H, W]
            else:
                # Shape: [H, W, C]
                img = np.transpose(img, (2, 0, 1))  # Shape: [C, H, W]
        else:
            raise ValueError(f"Unexpected img.ndim after slicing: {img.ndim}")

        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]  # Shape: [1, H, W]
        elif mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask.squeeze(-1)  # Shape: [H, W]
                mask = mask[np.newaxis, ...]  # Shape: [1, H, W]
            else:
                mask = np.transpose(mask, (2, 0, 1))  # Shape: [C, H, W]
        else:
            raise ValueError(f"Unexpected mask.ndim after slicing: {mask.ndim}")

        # Convert to NumPy arrays of type float32 and int64
        img = img.astype(np.float32)
        mask = mask.astype(np.int64)

        # Apply Albumentations transforms if provided
        if self.transform:
            # Albumentations expects images in [H, W, C] and masks in [H, W]
            img_np = img.transpose(1, 2, 0)  # [H, W, C]
            mask_np = mask.squeeze(0)        # [H, W]
            augmented = self.transform(image=img_np, mask=mask_np)
            img = augmented['image']
            mask = augmented['mask']
            # Transpose image back to [C, H, W] and add channel dimension to mask
            img = img.transpose(2, 0, 1)  # [C, H, W]
            mask = mask[np.newaxis, ...]  # [1, H, W]
            # Convert to tensors
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()
        else:
            # Convert to tensors
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()

        # Resize to 256x256
        img = F.interpolate(img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=(256, 256), mode='nearest').squeeze(0).long()

        return img, mask

def get_training_augmentation():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5,
            border_mode=0
        ),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    ])
    return train_transform

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        # self.bn = nn.BatchNorm2d(out_c)
        self.bn = nn.GroupNorm(num_groups=32, num_channels=out_c)
        # self.bn = nn.InstanceNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Initialize weights
        for m in self.se.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x * self.se(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_block1 = Conv2D(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        self.atrous_block6 = Conv2D(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = Conv2D(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = Conv2D(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv2D(in_channels, out_channels, kernel_size=1, padding=0)
        )
        self.conv1 = Conv2D(out_channels * 5, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = self.avg_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv1(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, in_channels=3):
        super().__init__()
        self.vit = vit_base_patch16_224(pretrained=True)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Modify the input projection layer
        if in_channels != 3:
            self.vit.patch_embed.proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
            nn.init.kaiming_normal_(self.vit.patch_embed.proj.weight, mode='fan_out', nonlinearity='relu')
            if self.vit.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.vit.patch_embed.proj.bias)

    def forward(self, x):
        # Resize input
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        # Extract features
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        # Exclude class token and reshape
        x = x[:, 1:, :].transpose(1, 2)
        h = w = self.img_size // self.patch_size
        x = x.view(-1, self.embed_dim, h, w)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = Conv2D(in_c, out_c)
        self.conv2 = Conv2D(out_c, out_c)
        self.se_block = SqueezeExcitationBlock(out_c)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se_block(x)
        return x

class Encoder1(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        network = vgg19(pretrained=True)
        features = list(network.features)

        # Modify the first convolutional layer
        first_conv_layer = features[0]
        new_first_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=first_conv_layer.bias is not None
        )

        # Handle weight initialization
        with torch.no_grad():
            if in_channels == 3:
                # Same number of channels; copy weights directly
                new_first_layer.weight.data.copy_(first_conv_layer.weight.data)
            elif in_channels > 3:
                # More channels; copy existing weights and initialize extra channels
                new_first_layer.weight.data[:, :3, :, :] = first_conv_layer.weight.data
                nn.init.kaiming_normal_(new_first_layer.weight.data[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            elif in_channels == 1:
                # Single channel; average the weights across the RGB channels
                averaged_weights = first_conv_layer.weight.data.mean(dim=1, keepdim=True)
                new_first_layer.weight.data.copy_(averaged_weights)
            else:
                # Fewer than 3 channels; copy the corresponding number of channels
                new_first_layer.weight.data.copy_(first_conv_layer.weight.data[:, :in_channels, :, :])

            # Copy bias if it exists
            if first_conv_layer.bias is not None:
                new_first_layer.bias.data.copy_(first_conv_layer.bias.data)

        features[0] = new_first_layer
        self.features = nn.Sequential(*features)

    def forward(self, x):
        skip_connections = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in [4, 9, 18, 27]:
                skip_connections.append(x)
        return x, skip_connections[::-1]

class Decoder1(nn.Module):
    def __init__(self, skip_channels):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(512 + skip_channels[0], 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(256 + skip_channels[1], 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(128 + skip_channels[2], 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(64 + skip_channels[3], 64)

    def forward(self, x, skips):
        x = self.up1(x)
        x = torch.cat([x, skips[0]], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, skips[1]], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, skips[2]], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, skips[3]], dim=1)
        x = self.conv4(x)
        return x

class Encoder2(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []

        x1 = self.conv1(x)
        skip_connections.append(x1)
        x = self.pool1(x1)

        x2 = self.conv2(x)
        skip_connections.append(x2)
        x = self.pool2(x2)

        x3 = self.conv3(x)
        skip_connections.append(x3)
        x = self.pool3(x3)

        x4 = self.conv4(x)
        skip_connections.append(x4)
        x = self.pool4(x4)

        return x, skip_connections[::-1]

class Decoder2(nn.Module):
    def __init__(self, skip_channels1, skip_channels2):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(512 + skip_channels1[0] + skip_channels2[0], 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(256 + skip_channels1[1] + skip_channels2[1], 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(128 + skip_channels1[2] + skip_channels2[2], 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(64 + skip_channels1[3] + skip_channels2[3], 64)

    def forward(self, x, skip1, skip2):
        x = self.up1(x)
        skip1_0 = F.interpolate(skip1[0], size=x.shape[2:], mode='bilinear', align_corners=False)
        skip2_0 = skip2[0]
        x = torch.cat([x, skip1_0, skip2_0], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        skip1_1 = F.interpolate(skip1[1], size=x.shape[2:], mode='bilinear', align_corners=False)
        skip2_1 = skip2[1]
        x = torch.cat([x, skip1_1, skip2_1], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        skip1_2 = F.interpolate(skip1[2], size=x.shape[2:], mode='bilinear', align_corners=False)
        skip2_2 = skip2[2]
        x = torch.cat([x, skip1_2, skip2_2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        skip1_3 = F.interpolate(skip1[3], size=x.shape[2:], mode='bilinear', align_corners=False)
        skip2_3 = skip2[3]
        x = torch.cat([x, skip1_3, skip2_3], dim=1)
        x = self.conv4(x)
        return x

class BuildDoubleUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, 
                 use_aspp=True, use_se=True, use_vit=True, 
                 vit_for_first=True, vit_for_second=True): 
        """
        Double U-Net with optional ASPP, SE, and ViT blocks.
        
        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output channels.
            use_aspp (bool): Whether to use ASPP blocks.
            use_se (bool): Whether to use Squeeze-Excitation blocks.
            use_vit (bool): Whether to use Vision Transformer.
            vit_for_first (bool): Use ViT in the first U-Net.
            vit_for_second (bool): Use ViT in the second U-Net.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_aspp = use_aspp
        self.use_se = use_se
        self.use_vit = use_vit
        self.vit_for_first = vit_for_first
        self.vit_for_second = vit_for_second

        # First U-Net components
        self.e1 = Encoder1(in_channels=in_channels)
        if self.use_aspp:
            self.a1 = ASPP(512, 512)
        self.d1 = None  # Initialized later based on skip channels
        self.outc1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # ViT block for the first U-Net
        if self.use_vit and self.vit_for_first:
            self.vit1 = ViTBlock(img_size=224, patch_size=16, embed_dim=768, in_channels=in_channels)
            self.vit_proj1 = nn.Conv2d(768, 512, kernel_size=1)

        # Second U-Net components
        self.e2 = Encoder2(in_channels=in_channels)
        if self.use_aspp:
            self.a2 = ASPP(512, 512)
        self.d2 = None  # Initialized later based on skip channels
        self.outc2 = nn.Conv2d(64, num_classes, kernel_size=1)

        # ViT block for the second U-Net
        if self.use_vit and self.vit_for_second:
            self.vit2 = ViTBlock(img_size=224, patch_size=16, embed_dim=768, in_channels=in_channels)
            self.vit_proj2 = nn.Conv2d(768, 512, kernel_size=1)

    def forward(self, x):
        # First U-Net
        x1, skip1 = self.e1(x)
        skip_channels1 = [s.size(1) for s in skip1]
        if self.d1 is None:
            self.d1 = Decoder1(skip_channels1).to(x.device)
        if self.use_aspp:
            x1 = self.a1(x1)
        if self.use_vit and self.vit_for_first:
            vit_features1 = self.vit1(x)
            vit_features_proj1 = self.vit_proj1(vit_features1)
            x1 = x1 + F.interpolate(vit_features_proj1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.d1(x1, skip1)
        y1 = self.outc1(x1)

        # Prepare input for second U-Net
        if self.num_classes == 1:
            y1_sigmoid = self.sigmoid(y1)
        else:
            y1_softmax = torch.softmax(y1, dim=1)
            y1_sigmoid = torch.sum(y1_softmax[:, 1:, :, :], dim=1, keepdim=True)

        y1_sigmoid_upsampled = F.interpolate(y1_sigmoid, size=x.shape[2:], mode='bilinear', align_corners=False)
        x2_input = x * y1_sigmoid_upsampled

        # Second U-Net
        x2, skip2 = self.e2(x2_input)
        skip_channels2 = [s.size(1) for s in skip2]
        if self.d2 is None:
            self.d2 = Decoder2(skip_channels1, skip_channels2).to(x.device)
        if self.use_aspp:
            x2 = self.a2(x2)
        if self.use_vit and self.vit_for_second:
            vit_features2 = self.vit2(x2_input)
            vit_features_proj2 = self.vit_proj2(vit_features2)
            x2 = x2 + F.interpolate(vit_features_proj2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x2 = self.d2(x2, skip1, skip2)
        y2 = self.outc2(x2)
        return y1, y2

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, num_classes, save_path, task, use_aspp, use_se, use_vit, vit_for_first, vit_for_second, use_albumentations):
    best_loss = float('inf')
    scaler = GradScaler()

    metrics_history = {'train': {}, 'val': {}}
    for phase in ['train', 'val']:
        # metrics_history[phase] = {'Loss': [], 'Dice': [], 'Jaccard': [], 'Precision': [], 'Recall': [], 'F2': []}
        metrics_history[phase] = {'Loss': [], 'Dice': [], 'Jaccard': [], 'Precision': [], 'Recall': [], 'F2': [], 'IoU': [], 'Hausdorff': []}


    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            batch_losses = []
            # batch_metrics = {'Dice': [], 'Jaccard': [], 'Precision': [], 'Recall': [], 'F2': []}
            batch_metrics = {'Dice': [], 'Jaccard': [], 'Precision': [], 'Recall': [], 'F2': [], 'IoU': [], 'Hausdorff': []}

            for inputs, masks in tqdm(dataloaders[phase], desc=f'{phase}'):
                inputs = inputs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                with autocast():
                    y1_pred, y2_pred = model(inputs)
                    if num_classes == 1:
                        masks = masks.float()
                        loss = criterion(y2_pred, masks)
                    else:
                        masks = masks.squeeze(1).long()
                        loss = criterion(y2_pred, masks)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                batch_losses.append(loss.item())

                with torch.no_grad():
                    if num_classes == 1:
                        y_pred = torch.sigmoid(y2_pred)
                        y_pred = (y_pred > 0.5).float()
                        dice = dice_score(masks.view(-1), y_pred.view(-1))
                        jaccard = jac_score(masks.view(-1), y_pred.view(-1))
                        precision_val = precision(masks.view(-1), y_pred.view(-1))
                        recall_val = recall(masks.view(-1), y_pred.view(-1))
                        f2_val = F2(masks.view(-1), y_pred.view(-1))
                        iou = iou_score(y_pred, masks)
                        hd_value = hausdorff_distance(y_pred.squeeze(1), masks.squeeze(1))
                    else:
                        y_pred = y2_pred
                        dice = dice_score_multiclass(masks, y_pred, num_classes)
                        jaccard = jac_score_multiclass(masks, y_pred, num_classes)
                        precision_val = precision_multiclass(masks, y_pred, num_classes)
                        recall_val = recall_multiclass(masks, y_pred, num_classes)
                        f2_val = F2_multiclass(masks, y_pred, num_classes)
                        iou = iou_score_multiclass(y_pred, masks, num_classes)
                        hd_values = []
                        for cls in range(1, num_classes):
                            y_pred_cls = torch.argmax(y_pred, dim=1) == cls
                            masks_cls = masks == cls
                            hd_value_cls = hausdorff_distance(y_pred_cls.float(), masks_cls.float())
                            hd_values.append(hd_value_cls)
                        hd_value = np.nanmean(hd_values)
                        

                    batch_metrics['Dice'].append(dice.item())
                    batch_metrics['Jaccard'].append(jaccard.item())
                    batch_metrics['Precision'].append(precision_val.item())
                    batch_metrics['Recall'].append(recall_val.item())
                    batch_metrics['F2'].append(f2_val.item())
                    batch_metrics['IoU'].append(iou.item())
                    batch_metrics['Hausdorff'].append(hd_value)

            epoch_loss = np.mean(batch_losses)
            metrics_history[phase]['Loss'].append(epoch_loss)
            for metric in batch_metrics:
                metrics_history[phase][metric].append(np.mean(batch_metrics[metric]))

            print(f'{phase} Loss: {epoch_loss:.4f}')
            print(f"{phase} Metrics: " + ", ".join([f"{metric}: {metrics_history[phase][metric][-1]:.4f}" for metric in batch_metrics]))

            # Save the model if it has the best validation loss so far
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                create_dir('models')
                # torch.save(model.state_dict(), f'{save_path}/{task}_best_model_albu_{use_albumentations}.pth')
                torch.save(model, f'{save_path}/best_model_aspp_{use_aspp}_se_{use_se}_vit_{use_vit}_vit1st_{vit_for_first}_vit2nd_{vit_for_second}_albu_{use_albumentations}.pth')
                

    print('Training complete')
    print(f'Best val Loss: {best_loss:.4f}')
    return model, metrics_history

def main(task, use_aspp, use_se, use_vit, vit_for_first, vit_for_second, use_albumentations):
    seeding(42)
    
    save_path = f'./abalation_study/{task}/train'
    create_dir(save_path)
    
    # use_albumentations = False

    config = task_configs[task]
    modalities = config['modalities']
    in_channels = config['in_channels']
    num_classes = config['num_classes']
    loss_function = config['loss_function']
    slice_axis = config['slice_axis']

    # Convert modalities to indices if they are strings
    if modalities is not None and isinstance(modalities[0], str):
        # Assuming modalities are ordered and you know their indices
        # For example, mapping modality names to indices
        modality_name_to_index = {
            'FLAIR': 0,
            'T1w': 1,
            'T1gd': 2,
            'T2w': 3,
            'MRI': 0,
            'CT': 0,
            'T2': 0,
            'ADC': 1
            # Add other modality mappings as needed
        }
        modalities = [modality_name_to_index[m] for m in modalities]

    # Dataset paths
    image_dir = os.path.join('./dataset', task, 'imagesTr')
    mask_dir = os.path.join('./dataset', task, 'labelsTr')

    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.nii.gz')))

    # Train-Val split
    # train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
    train_val_image_paths, test_image_paths, train_val_mask_paths, test_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=42)
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
        train_val_image_paths, train_val_mask_paths, test_size=0.1875, random_state=42)

    # Define transforms
    train_transform = get_training_augmentation() if use_albumentations else None
    
    # Create Datasets
    train_dataset = MSDDataset(
        train_image_paths, train_mask_paths,
        modalities=modalities, slice_axis=slice_axis,
        transform=train_transform
    )
    val_dataset = MSDDataset(
        val_image_paths, val_mask_paths,
        modalities=modalities, slice_axis=slice_axis,
        transform=None  # No augmentation for validation
    )
    # Create Datasets
    # train_dataset = MSDDataset(train_image_paths, train_mask_paths, modalities=modalities, slice_axis=slice_axis)
    # val_dataset = MSDDataset(val_image_paths, val_mask_paths, modalities=modalities, slice_axis=slice_axis)

    # Create Dataloaders
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f'Using device: {device}')
    # model = BuildDoubleUNet(in_channels=in_channels, num_classes=num_classes).to(device)
    # Initialize model
    model = BuildDoubleUNet(
        in_channels=in_channels, 
        num_classes=num_classes,
        use_aspp=use_aspp, 
        use_se=use_se, 
        use_vit=use_vit,
        vit_for_first=vit_for_first, 
        vit_for_second=vit_for_second
    ).to(device)

    # Define loss function
    loss_functions = {
        'CrossEntropy': nn.CrossEntropyLoss(),
        'BCEWithLogits': nn.BCEWithLogitsLoss(),
        'DiceLoss': DiceLoss(),
        'DiceBCELoss': DiceBCELoss(),
        'DiceLossMultiClass': DiceLossMultiClass()
    }
    criterion = loss_functions.get(loss_function)
    if criterion is None:
        raise ValueError('Invalid loss function specified')

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Define scheduler (optional)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Train the model
    print(f"Training the model on {task}....")
    model, metrics_history = train_model(
        model, dataloaders, criterion, optimizer,
        num_epochs=100, device=device, num_classes=num_classes,
        save_path=save_path, task=task, use_aspp=use_aspp, 
        use_se=use_se, 
        use_vit=use_vit,
        vit_for_first=vit_for_first, 
        vit_for_second=vit_for_second,
        use_albumentations=use_albumentations
    )

    print(metrics_history)

    def save_metrics_to_csv(metrics_history, filename):
        # Flatten the metrics_history dictionary
        data = []
        for phase, metrics in metrics_history.items():
            for metric_name, values in metrics.items():
                for epoch, value in enumerate(values):
                    data.append({"Phase": phase, "Metric": metric_name, "Epoch": epoch, "Value": value})

        # Convert to a DataFrame and save as CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Metrics history saved to {filename}")
    
    save_metrics_to_csv(metrics_history, f"{save_path}/metrics_aspp_{use_aspp}_se_{use_se}_vit_{use_vit}_vit1st_{vit_for_first}_vit2nd_{vit_for_second}_albu_{use_albumentations}.csv")

    # Save metrics
    with open(f'{save_path}/metrics_aspp_{use_aspp}_se_{use_se}_vit_{use_vit}_vit1st_{vit_for_first}_vit2nd_{vit_for_second}_albu_{use_albumentations}.pkl', 'wb') as f:
        pickle.dump(metrics_history, f)

    # Plot metrics
    # metric_names = ['Loss', 'Dice', 'Jaccard', 'Precision', 'Recall', 'F2']
    # plot_metrics(metrics_history, metric_names)
    # plt.savefig(f'./results_new/{task}_metrics_plot.png')
    # plt.close()

if __name__ == '__main__':
    # Baseline U-Net
    # main(task='Task01_BrainTumour', use_aspp=False, use_se=False, use_vit=False, vit_for_first=False, vit_for_second=False, use_albumentations=False) # Done
    # main(task='Task01_BrainTumour', use_aspp=False, use_se=False, use_vit=False, vit_for_first=False, vit_for_second=False, use_albumentations=True) #Done

    # # Double U-Net without ASPP
    # main(task='Task01_BrainTumour', use_aspp=False, use_se=True, use_vit=False, vit_for_first=False, vit_for_second=False, use_albumentations=True) # Done
    # main(task='Task01_BrainTumour', use_aspp=False, use_se=True, use_vit=False, vit_for_first=False, vit_for_second=False, use_albumentations=False) # Done

    # # # Double U-Net with ASPP
    # main(task='Task01_BrainTumour', use_aspp=True, use_se=True, use_vit=False, vit_for_first=False, vit_for_second=False, use_albumentations=True) # done
    # main(task='Task01_BrainTumour', use_aspp=True, use_se=True, use_vit=False, vit_for_first=False, vit_for_second=False, use_albumentations=False) # done
    
    # # # Double U-Net + ViT on First U-Net
    # main(task='Task01_BrainTumour', use_aspp=True, use_se=True, use_vit=True, vit_for_first=True, vit_for_second=False, use_albumentations=True) #Done
    main(task='Task01_BrainTumour', use_aspp=True, use_se=True, use_vit=True, vit_for_first=True, vit_for_second=False, use_albumentations=False) # running

    # # # Full Model with ViT on Both U-Nets
    # main(task='Task01_BrainTumour', use_aspp=True, use_se=True, use_vit=True, vit_for_first=True, vit_for_second=True, use_albumentations=False) #Done
    # main(task='Task01_BrainTumour', use_aspp=True, use_se=True, use_vit=True, vit_for_first=True, vit_for_second=True, use_albumentations=True) #Done
