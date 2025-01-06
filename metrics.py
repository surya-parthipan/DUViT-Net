import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial.distance import directed_hausdorff
import numpy as np

""" Loss Functions -------------------------------------- """
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class DiceLossMultiClass(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLossMultiClass, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        # Apply softmax to inputs
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets = torch.eye(inputs.shape[1])[targets.squeeze(1)].permute(0, 3, 1, 2).to(inputs.device)
        
        # Flatten inputs and targets
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

""" Metrics ------------------------------------------ """
def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def dice_score_3d(preds, targets, smooth=1e-5):
    preds = (torch.sigmoid(preds) > 0.5).float()
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice
    
def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def iou_score(pred, target, smooth=1e-6):
    """
    Computes the Intersection over Union (IoU) score.

    Parameters:
    - pred (Tensor): Predicted mask of shape [N, H, W] or [N, 1, H, W].
    - target (Tensor): Ground truth mask of shape [N, H, W] or [N, 1, H, W].
    - smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    - float: IoU score.
    """
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    total = (pred + target).sum()
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou

def dice_score_multiclass(y_true, y_pred, num_classes):
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = y_true.squeeze(1)
    
    dice = 0
    for cls in range(num_classes):
        y_true_cls = (y_true == cls).float()
        y_pred_cls = (y_pred == cls).float()
        intersection = (y_true_cls * y_pred_cls).sum()
        union = y_true_cls.sum() + y_pred_cls.sum()
        dice += (2. * intersection + 1e-15) / (union + 1e-15)
    return dice / num_classes

def precision_multiclass(y_true, y_pred, num_classes):
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = y_true.squeeze(1)
    
    precision = 0
    for cls in range(num_classes):
        y_true_cls = (y_true == cls).float()
        y_pred_cls = (y_pred == cls).float()
        tp = (y_true_cls * y_pred_cls).sum()
        fp = (y_pred_cls * (1 - y_true_cls)).sum()
        precision += (tp + 1e-15) / (tp + fp + 1e-15)
    return precision / num_classes

def recall_multiclass(y_true, y_pred, num_classes):
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = y_true.squeeze(1)
    
    recall = 0
    for cls in range(num_classes):
        y_true_cls = (y_true == cls).float()
        y_pred_cls = (y_pred == cls).float()
        tp = (y_true_cls * y_pred_cls).sum()
        fn = (y_true_cls * (1 - y_pred_cls)).sum()
        recall += (tp + 1e-15) / (tp + fn + 1e-15)
    return recall / num_classes

def F2_multiclass(y_true, y_pred, num_classes, beta=2):
    p = precision_multiclass(y_true, y_pred, num_classes)
    r = recall_multiclass(y_true, y_pred, num_classes)
    return (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-15)

def jac_score_multiclass(y_true, y_pred, num_classes):
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = y_true.squeeze(1)
    
    jaccard = 0
    for cls in range(num_classes):
        y_true_cls = (y_true == cls).float()
        y_pred_cls = (y_pred == cls).float()
        intersection = (y_true_cls * y_pred_cls).sum()
        union = y_true_cls.sum() + y_pred_cls.sum() - intersection
        jaccard += (intersection + 1e-15) / (union + 1e-15)
    return jaccard / num_classes

def iou_score_multiclass(pred, target, num_classes, smooth=1e-6):
    """
    Computes the mean Intersection over Union (IoU) score for multi-class segmentation.

    Parameters:
    - pred (Tensor): Predicted logits or probabilities of shape [N, C, H, W].
    - target (Tensor): Ground truth mask of shape [N, H, W].
    - num_classes (int): Number of classes.
    - smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    - float: Mean IoU score across all classes.
    """
    pred = torch.argmax(pred, dim=1)  # Shape: [N, H, W]
    iou_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum().item()
        union = pred_cls.sum().item() + target_cls.sum().item() - intersection

        if union == 0:
            iou = np.nan  # Avoid division by zero
        else:
            iou = (intersection + smooth) / (union + smooth)
        iou_list.append(iou)
        
    return np.nanmean(iou_list)


import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(pred, target):
    """
    Computes the Hausdorff Distance between two binary masks.

    Parameters:
    - pred (Tensor): Predicted mask of shape [H, W].
    - target (Tensor): Ground truth mask of shape [H, W].

    Returns:
    - float: Hausdorff Distance.
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    # Ensure binary
    pred = (pred > 0).astype(np.bool_)
    target = (target > 0).astype(np.bool_)

    if np.count_nonzero(pred) == 0 or np.count_nonzero(target) == 0:
        return np.nan  # Undefined when one of the masks is empty

    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)

    # Compute the directed Hausdorff distances
    forward_hd = directed_hausdorff(pred_points, target_points)[0]
    backward_hd = directed_hausdorff(target_points, pred_points)[0]

    hd = max(forward_hd, backward_hd)
    return hd


def hd95(pred, gt, voxelspacing=None):
    """
    Compute the 95th percentile of the Hausdorff Distance between binary objects in pred and gt.

    Parameters:
    - pred (Tensor): Binary tensor of predicted mask.
    - gt (Tensor): Binary tensor of ground truth mask.
    - voxelspacing (float or sequence of floats, optional): Spacing of elements along each dimension.

    Returns:
    - float: The 95th percentile of the Hausdorff Distance.
    """
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    # Ensure binary
    pred = (pred > 0).astype(np.bool_)
    gt = (gt > 0).astype(np.bool_)

    if np.count_nonzero(pred) == 0 or np.count_nonzero(gt) == 0:
        return np.nan  # Undefined when one of the masks is empty

    # Compute the distance from the border of pred to the border of gt
    dt_gt = distance_transform_edt(~gt, sampling=voxelspacing)
    sds_pred = dt_gt[pred]

    dt_pred = distance_transform_edt(~pred, sampling=voxelspacing)
    sds_gt = dt_pred[gt]

    hd95_value = np.percentile(np.hstack((sds_pred, sds_gt)), 95)
    
    return hd95_value


def normalized_surface_dice(y_pred, y_true, spacing, tolerance=1):
    """
    Computes the Normalized Surface Dice (NSD) between the predicted and ground truth masks.

    Args:
        y_pred (numpy.ndarray): Predicted binary mask. Shape: (H, W)
        y_true (numpy.ndarray): Ground truth binary mask. Shape: (H, W)
        spacing (tuple or list): Voxel spacing along each axis.
        tolerance (float): Tolerance distance in millimeters.

    Returns:
        nsd (float): The NSD score.
    """
    # Ensure the inputs are binary masks of type bool
    y_pred = (y_pred > 0.5)
    y_true = (y_true > 0.5)

    # Squeeze to remove singleton dimensions
    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)

    # Check shapes and dimensions
    print(f"Inside normalized_surface_dice:")
    print(f"y_pred shape: {y_pred.shape}, ndim: {y_pred.ndim}, dtype: {y_pred.dtype}")
    print(f"y_true shape: {y_true.shape}, ndim: {y_true.ndim}, dtype: {y_true.dtype}")

    # Define structure based on input dimensionality
    structure = np.ones((3,) * y_pred.ndim)

    # Perform binary erosion
    pred_eroded = binary_erosion(y_pred, structure=structure)
    true_eroded = binary_erosion(y_true, structure=structure)

    # Extract surface voxels
    pred_surface = y_pred ^ pred_eroded
    true_surface = y_true ^ true_eroded

    # Compute distance maps
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    dt_true = distance_transform_edt(~true_surface, sampling=spacing)

    # Surface distances
    pred_to_true = dt_true[pred_surface]
    true_to_pred = dt_pred[true_surface]

    # Count surface voxels within tolerance
    num_pred_close = np.sum(pred_to_true <= tolerance)
    num_true_close = np.sum(true_to_pred <= tolerance)

    # Total number of surface voxels
    num_pred_surface = np.sum(pred_surface)
    num_true_surface = np.sum(true_surface)

    # Compute NSD
    nsd = (num_pred_close + num_true_close) / (num_pred_surface + num_true_surface + 1e-8)

    return nsd


