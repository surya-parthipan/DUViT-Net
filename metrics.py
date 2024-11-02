import torch
import torch.nn as nn
import torch.nn.functional as F

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

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

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


# Function to calculate overall metrics
def calculate_overall_metrics(metrics_history):
    overall_metrics = {}

    for phase in metrics_history:
        overall_metrics[phase] = {}

        # Iterate over each metric in that phase
        for metric, values in metrics_history[phase].items():
            # Extract the values across epochs (ignore the std stored per epoch here)
            metric_values = [v[0] for v in values]  # Mean values across epochs
            metric_stds = [v[1] for v in values]    # Std dev values across epochs

            # Compute mean and std dev for the metric across all epochs
            overall_mean = np.mean(metric_values)
            overall_std = np.std(metric_values)

            # Store the overall metrics
            overall_metrics[phase][metric] = {
                'mean': overall_mean,
                'std_dev': overall_std
            }

    return overall_metrics