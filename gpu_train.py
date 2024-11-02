import os
import glob
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import build_doubleunet  
from dataset import MSDDataset 
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import VGG19_BN_Weights
import matplotlib.pyplot as plt
from utils import seeding, create_dir, plot_metrics
from metrics import (DiceLoss, DiceBCELoss, DiceLossMultiClass, precision, recall, F2, dice_score, jac_score, precision_multiclass, recall_multiclass, F2_multiclass, dice_score_multiclass, jac_score_multiclass, calculate_overall_metrics)
import pickle

import warnings
warnings.filterwarnings("ignore")

# Load task config
with open('tasks_config.yaml', 'r') as f:
    task_configs = yaml.safe_load(f)

# Training function with mixed precision
def train_model(model, dataloaders, criterion, optimizer, num_epochs=100, device='cuda', num_classes=1):
    best_loss = float('inf')
    scaler = GradScaler()  # Initialize GradScaler for mixed precision # key updates

    # Init dic to store metrics
    metrics_history = {
        'train': {
            'Loss': [],
            'Dice': [],
            'Jaccard': [],
            'Precision': [],
            'Recall': [],
            'F2': [],
        },
        'val': {
            'Loss': [],
            'Dice': [],
            'Jaccard': [],
            'Precision': [],
            'Recall': [],
            'F2': [],
        }
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Training mode
            else:
                model.eval()   # Eval mode

            batch_losses = []
            batch_dices = []
            batch_jaccards = []
            batch_precisions = []
            batch_recalls = []
            batch_f2s = []

            # Iterate over data
            for inputs, masks in tqdm(dataloaders[phase], desc=f'{phase}'):
                inputs = inputs.to(device)
                masks = masks.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Mixed precision training
                with autocast():
                    # Forward pass
                    y1_pred, y2_pred = model(inputs)
                    # Use the final output y2_pred
                    if num_classes == 1:
                        masks = masks.float()
                        loss = criterion(y2_pred, masks)
                    else:
                        masks = masks.squeeze(1).long()
                        loss = criterion(y2_pred, masks)

                # Backward + optimize only in training phase
                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # Record batch loss
                batch_losses.append(loss.item())

                with torch.no_grad():
                    if num_classes == 1:
                        # For binary segmentation
                        y_pred = torch.sigmoid(y2_pred)
                        y_pred = (y_pred > 0.5).float()

                        dice = dice_score(masks.view(-1), y_pred.view(-1))
                        batch_dices.append(dice.item())

                        precision_val = precision(masks.view(-1), y_pred.view(-1))
                        recall_val = recall(masks.view(-1), y_pred.view(-1))
                        f2_val = F2(masks.view(-1), y_pred.view(-1))
                        jaccard_val = jac_score(masks.view(-1), y_pred.view(-1))

                    else:
                        # For multi-class segmentation
                        dice = dice_score_multiclass(masks, y2_pred, num_classes)
                        batch_dices.append(dice.item())

                        precision_val = precision_multiclass(masks, y2_pred, num_classes)
                        recall_val = recall_multiclass(masks, y2_pred, num_classes)
                        f2_val = F2_multiclass(masks, y2_pred, num_classes)
                        jaccard_val = jac_score_multiclass(masks, y2_pred, num_classes)

                    batch_precisions.append(precision_val.item())
                    batch_recalls.append(recall_val.item())
                    batch_f2s.append(f2_val.item())
                    batch_jaccards.append(jaccard_val.item())

            # Compute epoch statistics
            epoch_loss = np.mean(batch_losses)
            epoch_loss_std = np.std(batch_losses)

            epoch_dice = np.mean(batch_dices)
            epoch_dice_std = np.std(batch_dices)

            epoch_jaccard = np.mean(batch_jaccards)
            epoch_jaccard_std = np.std(batch_jaccards)

            epoch_precision = np.mean(batch_precisions)
            epoch_precision_std = np.std(batch_precisions)

            epoch_recall = np.mean(batch_recalls)
            epoch_recall_std = np.std(batch_recalls)

            epoch_f2 = np.mean(batch_f2s)
            epoch_f2_std = np.std(batch_f2s)

            # Store the epoch metrics
            metrics_history[phase]['Loss'].append((epoch_loss, epoch_loss_std))
            metrics_history[phase]['Dice'].append((epoch_dice, epoch_dice_std))
            metrics_history[phase]['Jaccard'].append((epoch_jaccard, epoch_jaccard_std))
            metrics_history[phase]['Precision'].append((epoch_precision, epoch_precision_std))
            metrics_history[phase]['Recall'].append((epoch_recall, epoch_recall_std))
            metrics_history[phase]['F2'].append((epoch_f2, epoch_f2_std))

            # Print metrics
            print(f'{phase} Loss: {epoch_loss:.4f} ± {epoch_loss_std:.4f}')
            print(f'Dice: {epoch_dice:.4f} ± {epoch_dice_std:.4f}')
            print(f'Jaccard: {epoch_jaccard:.4f} ± {epoch_jaccard_std:.4f}')
            print(f'Precision: {epoch_precision:.4f} ± {epoch_precision_std:.4f}')
            print(f'Recall: {epoch_recall:.4f} ± {epoch_recall_std:.4f}')
            print(f'F2 Score: {epoch_f2:.4f} ± {epoch_f2_std:.4f}')

            # Save the model if it has the best validation loss so far
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                create_dir('models')  # Use create_dir from utils.py
                torch.save(model.state_dict(), 'models/best_model.pth')

    print('Training complete')
    print(f'Best val Loss: {best_loss:.4f}')
    return model, metrics_history

def main():
    seeding(42)
    task = 'Task01_BrainTumour' # Task to train

    config = task_configs[task]
    modalities = config['modalities']
    in_channels = config['in_channels']
    num_classes = config['num_classes']
    loss_function = config['loss_function']
    slice_axis = config['slice_axis']

    # dataset paths
    image_dir = f'./dataset/{task}/imagesTr'  
    mask_dir = f'./dataset/{task}/labelsTr'   

    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.nii.gz')))

    # Train-Val split
    val_split = 0.2
    num_images = len(image_paths)
    indices = list(range(num_images))
    split = int(np.floor(val_split * num_images))

    np.random.seed(42)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    train_image_paths = [image_paths[i] for i in train_idx]
    train_mask_paths = [mask_paths[i] for i in train_idx]
    val_image_paths = [image_paths[i] for i in val_idx]
    val_mask_paths = [mask_paths[i] for i in val_idx]

    # Create Datasets
    train_dataset = MSDDataset(train_image_paths, train_mask_paths, modalities=modalities, slice_axis=slice_axis)
    val_dataset = MSDDataset(val_image_paths, val_mask_paths, modalities=modalities, slice_axis=slice_axis)

    # Create Dataloaders
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}

    # Init model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = build_doubleunet(in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)

    # def loss function
    if loss_function == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'BCEWithLogits':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function == 'DiceLoss':
        criterion = DiceLoss()
    elif loss_function == 'DiceBCELoss':
        criterion = DiceBCELoss()
    elif loss_function == 'DiceLossMultiClass':
        criterion = DiceLossMultiClass()
    else:
        raise ValueError('Invalid loss function specified')

    # def optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model with mixed precision
    model, metrics_history = train_model(
        model, dataloaders, criterion, optimizer,
        num_epochs=100, device=device, num_classes=num_classes
    )

    with open(f'./results/{task}_metrics_history.pkl', 'wb') as f:
        pickle.dump(metrics_history, f)
        
    # Plotting
    metric_names = ['Loss', 'Dice', 'Jaccard', 'Precision', 'Recall', 'F2']
    plot_metrics(metrics_history, metric_names)
    plt.savefig(f'./results/{task}_metrics_plot.png')

if __name__ == '__main__':
    main()
