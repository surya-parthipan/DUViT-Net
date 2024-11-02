import os
import glob
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import build_doubleunet  # Ensure model code is updated
from dataset import MSDDataset  # Save the dataset class in dataset.py
import numpy as np
from tqdm import tqdm

# Load task configurations from tasks_config.yaml
with open('tasks_config.yaml', 'r') as f:
    task_configs = yaml.safe_load(f)

# Dice coefficient metric
def dice_coeff(pred, target, smooth=1e-5, num_classes=1):
    with torch.no_grad():
        if num_classes == 1:
            # Binary segmentation
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            intersection = (pred * target).sum()
            return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        else:
            # Multi-class segmentation
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            target = target.squeeze(1)
            intersection = (pred == target).float().sum()
            return (intersection + smooth) / (pred.numel() + smooth)

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda', num_classes=1):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Training mode
            else:
                model.eval()   # Evaluation mode

            running_loss = 0.0
            running_dice = 0.0

            # Iterate over data
            for inputs, masks in tqdm(dataloaders[phase], desc=f'{phase}'):
                inputs = inputs.to(device)
                masks = masks.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    y1_pred, y2_pred = model(inputs)
                    # Use the final output y2_pred
                    if num_classes == 1:
                        masks = masks.float()
                        loss = criterion(y2_pred, masks)
                    else:
                        masks = masks.squeeze(1).long()
                        loss = criterion(y2_pred, masks)

                    dice = dice_coeff(y2_pred, masks, num_classes=num_classes)

                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_dice += dice.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_dice = running_dice / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Dice: {epoch_dice:.4f}')

            # Save the model if it has the best validation loss so far
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'best_model.pth')

    print('Training complete')
    print(f'Best val Loss: {best_loss:.4f}')
    return model

def main():
    task = 'Task01_BrainTumour'

    config = task_configs[task]
    modalities = config['modalities']
    in_channels = config['in_channels']
    num_classes = config['num_classes']
    loss_function = config['loss_function']
    slice_axis = config['slice_axis']

    # Paths to images and masks
    image_dir = f'./dataset/{task}/imagesTr'
    mask_dir = f'./dataset/{task}/labelsTr'

    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.nii.gz')))

    # Split into train and val
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

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_doubleunet(in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)

    # Define loss function
    if loss_function == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'BCEWithLogits':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError('Invalid loss function specified')

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=50, device=device, num_classes=num_classes)

if __name__ == '__main__':
    main()
