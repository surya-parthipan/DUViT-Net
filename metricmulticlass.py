import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial.distance import directed_hausdorff
import numpy as np

def precision_multiclass(masks, y_pred, num_classes):
    precisions = {}
    for cls in range(num_classes):
        masks_cls = (masks == cls).float()
        y_pred_cls = (y_pred == cls).float()
        true_positive = (y_pred_cls * masks_cls).sum().item()
        predicted_positive = y_pred_cls.sum().item()
        if predicted_positive == 0:
            precisions[cls] = 1.0  # Assuming perfect precision when no positive predictions
        else:
            precisions[cls] = true_positive / predicted_positive
    return precisions

def recall_multiclass(masks, y_pred, num_classes):
    recalls = {}
    for cls in range(num_classes):
        masks_cls = (masks == cls).float()
        y_pred_cls = (y_pred == cls).float()
        true_positive = (y_pred_cls * masks_cls).sum().item()
        actual_positive = masks_cls.sum().item()
        if actual_positive == 0:
            recalls[cls] = 1.0  # Assuming perfect recall when no actual positives
        else:
            recalls[cls] = true_positive / actual_positive
    return recalls

def F2_multiclass(masks, y_pred, num_classes):
    f2_scores = {}
    for cls in range(num_classes):
        masks_cls = (masks == cls).float()
        y_pred_cls = (y_pred == cls).float()
        true_positive = (y_pred_cls * masks_cls).sum().item()
        predicted_positive = y_pred_cls.sum().item()
        actual_positive = masks_cls.sum().item()
        
        if predicted_positive == 0:
            precision = 1.0
        else:
            precision = true_positive / predicted_positive
        
        if actual_positive == 0:
            recall = 1.0
        else:
            recall = true_positive / actual_positive
        
        if precision + recall == 0:
            f2 = 0.0
        else:
            f2 = (5 * precision * recall) / (4 * precision + recall)
        
        f2_scores[cls] = f2
    return f2_scores

def iou_score_multiclass(y_pred, masks, num_classes):
    iou_scores = {}
    for cls in range(num_classes):
        y_pred_cls = (y_pred == cls).float()
        masks_cls = (masks == cls).float()
        intersection = (y_pred_cls * masks_cls).sum().item()
        union = y_pred_cls.sum().item() + masks_cls.sum().item() - intersection
        if union == 0:
            iou_scores[cls] = 1.0  # Perfect score if both are empty
        else:
            iou_scores[cls] = intersection / union
    return iou_scores

def jac_score_multiclass(masks, y_pred, num_classes):
    jac_scores = {}
    for cls in range(num_classes):
        masks_cls = (masks == cls).float()
        y_pred_cls = (y_pred == cls).float()
        intersection = (masks_cls * y_pred_cls).sum().item()
        union = masks_cls.sum().item() + y_pred_cls.sum().item() - intersection
        if union == 0:
            jac_scores[cls] = 1.0  # Perfect score if both are empty
        else:
            jac_scores[cls] = intersection / union
    return jac_scores


def dice_score_multiclass(masks, y_pred, num_classes):
    """
    Compute the Dice score for multi-class segmentation per class.

    Args:
        masks (torch.Tensor): Ground truth masks of shape [batch_size, H, W], with integer class labels.
        y_pred (torch.Tensor): Predicted masks of shape [batch_size, H, W], with integer class labels.
        num_classes (int): Number of classes.

    Returns:
        dict: Dice scores per class.
    """
    dice_scores = {}
    for cls in range(num_classes):
        masks_cls = (masks == cls).float()
        y_pred_cls = (y_pred == cls).float()
        intersection = (masks_cls * y_pred_cls).sum().item()
        union = masks_cls.sum().item() + y_pred_cls.sum().item()
        if union == 0:
            dice_scores[cls] = 1.0  # Perfect score if both are empty
        else:
            dice_scores[cls] = (2.0 * intersection) / union
    return dice_scores

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
    
def hausdorff_distance_multiclass(y_pred, masks, num_classes):
    hd_values = {}
    for cls in range(1, num_classes):  # Exclude background if class 0 is background
        y_pred_cls = (y_pred == cls).float()
        masks_cls = (masks == cls).float()
        hd_value = hausdorff_distance(y_pred_cls, masks_cls)
        hd_values[cls] = hd_value
    return hd_values
