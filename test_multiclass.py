import os
import glob
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import nibabel as nib
from torchvision.models import vgg19
from torchvision.models import resnet50
from timm.models.vision_transformer import vit_base_patch16_224
import albumentations as A
from utils import seeding, create_dir
from metrics import (DiceLoss, DiceBCELoss, DiceLossMultiClass, precision, recall, F2, dice_score, jac_score, hausdorff_distance)
    # precision_multiclass, recall_multiclass, F2_multiclass, dice_score_multiclass, jac_score_multiclass,
    # hausdorff_distance, iou_score, iou_score_multiclass, hd95, normalized_surface_dice)
from metricmulticlass import (precision_multiclass, recall_multiclass, F2_multiclass, dice_score_multiclass, jac_score_multiclass, hausdorff_distance_multiclass, iou_score_multiclass)
from sklearn.model_selection import train_test_split
from ablation import BuildDoubleUNet, Conv2D, SqueezeExcitationBlock, ASPP, ViTBlock, ConvBlock, Encoder1, Encoder2, Decoder1, Decoder2
# Load task config
with open('task_config.yaml', 'r') as f:
    task_configs = yaml.safe_load(f)

# MSDDataset class as defined in your training script
class MSDDataset(Dataset):
    # ... [Include the MSDDataset class code from your ablation study code here] ...
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
        # ... [Your __getitem__ method code here] ...
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
def evaluate_model(model, dataloader, criterion, device, num_classes):
    model.eval()
    batch_losses = []
    per_class_metrics = {
        'Dice': {cls: [] for cls in range(num_classes)},
        'Jaccard': {cls: [] for cls in range(num_classes)},
        'Precision': {cls: [] for cls in range(num_classes)},
        'Recall': {cls: [] for cls in range(num_classes)},
        'F2': {cls: [] for cls in range(num_classes)},
        'IoU': {cls: [] for cls in range(num_classes)},
        'Hausdorff': {cls: [] for cls in range(0, num_classes)},  # Exclude background if necessary
    }

    with torch.no_grad():
        for inputs, masks in tqdm(dataloader, desc='Testing'):
            inputs = inputs.to(device)
            masks = masks.to(device)

            y1_pred, y2_pred = model(inputs)
            # print(f"y2_pred Shape: {y2_pred.shape}")

            if num_classes == 1:
                masks = masks.float()
                loss = criterion(y2_pred, masks)
            else:
                masks = masks.squeeze(1).long()
                loss = criterion(y2_pred, masks)

            batch_losses.append(loss.item())

            if num_classes == 1:
                y_pred_probs = torch.sigmoid(y2_pred)
                y_pred = (y_pred_probs > 0.5).float()

                dice = dice_score(masks.view(-1), y_pred.view(-1))
                jaccard = jac_score(masks.view(-1), y_pred.view(-1))
                precision_val = precision(masks.view(-1), y_pred.view(-1))
                recall_val = recall(masks.view(-1), y_pred.view(-1))
                f2_val = F2(masks.view(-1), y_pred.view(-1))
                iou = iou_score(y_pred, masks)
                hd_value = hausdorff_distance(y_pred.squeeze(1), masks.squeeze(1))
            else:
                # Apply softmax and get predicted labels
                y_pred_probs = torch.softmax(y2_pred, dim=1)
                y_pred = torch.argmax(y_pred_probs, dim=1)  # Shape: [batch_size, H, W]

                # Compute per-class metrics
                dice_scores = dice_score_multiclass(masks, y_pred, num_classes)
                jaccard_scores = jac_score_multiclass(masks, y_pred, num_classes)
                precision_scores = precision_multiclass(masks, y_pred, num_classes)
                recall_scores = recall_multiclass(masks, y_pred, num_classes)
                f2_scores = F2_multiclass(masks, y_pred, num_classes)
                iou_scores = iou_score_multiclass(y_pred, masks, num_classes)
                hd_values = hausdorff_distance_multiclass(y_pred, masks, num_classes)

                # Aggregate per-class metrics
                for cls in range(num_classes):
                    per_class_metrics['Dice'][cls].append(dice_scores[cls])
                    per_class_metrics['Jaccard'][cls].append(jaccard_scores[cls])
                    per_class_metrics['Precision'][cls].append(precision_scores[cls])
                    per_class_metrics['Recall'][cls].append(recall_scores[cls])
                    per_class_metrics['F2'][cls].append(f2_scores[cls])
                    per_class_metrics['IoU'][cls].append(iou_scores[cls])
                for cls in range(1, num_classes):  # Exclude background if necessary
                    per_class_metrics['Hausdorff'][cls].append(hd_values[cls])

    test_loss = np.mean(batch_losses)

    # Compute mean per-class metrics
    mean_per_class_metrics = {}
    for metric in per_class_metrics:
        mean_per_class_metrics[metric] = {}
        for cls in per_class_metrics[metric]:
            values = per_class_metrics[metric][cls]
            mean_value = np.mean(values)
            mean_per_class_metrics[metric][cls] = mean_value

    # Print per-class metrics
    print(f'Test Loss: {test_loss:.4f}')
    print("Per-Class Metrics:")
    for cls in range(num_classes):
        print(f'Class {cls}:')
        for metric in mean_per_class_metrics:
            if cls in mean_per_class_metrics[metric]:
                value = mean_per_class_metrics[metric][cls]
                print(f'  {metric}: {value:.4f}')

    return test_loss, mean_per_class_metrics

def load_model(save_path, device, in_channels, num_classes, use_aspp, use_se, use_vit, vit_for_first, vit_for_second, use_albumentations):
    # Construct the model filename based on the configuration
    model_filename = f'best_model_aspp_{use_aspp}_se_{use_se}_vit_{use_vit}_vit1st_{vit_for_first}_vit2nd_{vit_for_second}_albu_{use_albumentations}.pth'
    model_path = os.path.join(save_path, model_filename)

    # Load the entire model object
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    return model
def main(task):
    seeding(42)

    save_path = f'./abalation_study/{task}/test'
    load_path = f'./abalation_study/{task}/train'
    create_dir(save_path)
    create_dir(load_path)

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

    # Load the test data paths
    # Use the same random_state as in training to get the same test set
    train_val_image_paths, test_image_paths, train_val_mask_paths, test_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=42)
    
    test_transform = get_training_augmentation()

    # Create Test Dataset
    test_dataset = MSDDataset(
        test_image_paths, test_mask_paths,
        modalities=modalities, slice_axis=slice_axis,
        # transform=None
        transform = test_transform
    )

    # Create Test DataLoader
    batch_size = 4
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

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

    # List of model configurations to evaluate
    configurations = [
        # {'use_aspp': False, 'use_se': False, 'use_vit': False, 'vit_for_first': False, 'vit_for_second': False, 'use_albumentations' : True},
        # {'use_aspp': False, 'use_se': False, 'use_vit': False, 'vit_for_first': False, 'vit_for_second': False, 'use_albumentations' : False},
        # {'use_aspp': False, 'use_se': True, 'use_vit': False, 'vit_for_first': False, 'vit_for_second': False, 'use_albumentations' : True},
        # {'use_aspp': False, 'use_se': True, 'use_vit': False, 'vit_for_first': False, 'vit_for_second': False, 'use_albumentations' : False},
        # {'use_aspp': True, 'use_se': True, 'use_vit': False, 'vit_for_first': False, 'vit_for_second': False, 'use_albumentations' : True},
        # {'use_aspp': True, 'use_se': True, 'use_vit': False, 'vit_for_first': False, 'vit_for_second': False, 'use_albumentations' : False},
        # {'use_aspp': True, 'use_se': True, 'use_vit': True, 'vit_for_first': True, 'vit_for_second': False, 'use_albumentations' : True},
        # {'use_aspp': True, 'use_se': True, 'use_vit': True, 'vit_for_first': True, 'vit_for_second': False, 'use_albumentations' : False},
        {'use_aspp': True, 'use_se': True, 'use_vit': True, 'vit_for_first': True, 'vit_for_second': True, 'use_albumentations' : True},
        # {'use_aspp': True, 'use_se': True, 'use_vit': True, 'vit_for_first': True, 'vit_for_second': True, 'use_albumentations' : False},
    ]
    
    evaluation_results = []

    for config in configurations:
        print(f"Evaluating model with configuration: {config}")
        # Load the model with the specified configuration
        model = load_model(
            save_path=load_path,
            device=device,
            in_channels=in_channels,
            num_classes=num_classes,
            use_aspp=config['use_aspp'],
            use_se=config['use_se'],
            use_vit=config['use_vit'],
            vit_for_first=config['vit_for_first'],
            vit_for_second=config['vit_for_second'],
            use_albumentations=config['use_albumentations']
        )

        # Evaluate the model on the test set
        test_loss, test_metrics = evaluate_model(
            model, test_loader, criterion, device, num_classes
        )

        # Store the results
        evaluation_results.append({
            'Configuration': config,
            'Test Loss': test_loss,
            **test_metrics
        })

    # Save the evaluation results to a CSV file
    df = pd.DataFrame(evaluation_results)
    results_filename = os.path.join(save_path, f'{task}_mulcls.csv')
    # results_filename = os.path.join(save_path, f'{task}_evaluation_results.csv')
    df.to_csv(results_filename, index=False)
    print(f"Evaluation results saved to {results_filename}")

    # Print the results
    print(df)

if __name__ == '__main__':
    task = 'Task01_BrainTumour'
    # task = 'Task02_Heart'
    # task = 'Task04_Hippocampus'
    main(task=task)
