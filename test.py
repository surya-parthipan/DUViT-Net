import os
import torch
import numpy as np
from tqdm import tqdm
import yaml
import pickle
import glob
from torch.utils.data import DataLoader
from model import build_doubleunet
from dataset import MSDDataset
from utils import seeding, create_dir, calculate_metrics

def process_mask(y_pred):
    """Process and binarize the predicted mask for visualization and metrics calculation."""
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = (y_pred > 0.5).astype(np.uint8) * 255
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred

def evaluate(model, dataloader, device):
    """Evaluate the model on the test dataset."""
    metrics_score = {"jaccard": 0, "dice": 0, "precision": 0, "recall": 0, "accuracy": 0, "f2": 0}
    time_taken = []

    for i, (inputs, masks) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
        inputs, masks = inputs.to(device), masks.to(device)

        with torch.no_grad():
            # Measure inference time
            start_time = time.time()
            y_pred1, y_pred2 = model(inputs)
            end_time = time.time() - start_time
            time_taken.append(end_time)

            # Use the final output y_pred2
            y_pred = torch.sigmoid(y_pred2) > 0.5
            y_pred = y_pred.float()

            # Calculate metrics
            metrics = calculate_metrics(masks, y_pred)
            for key in metrics_score:
                metrics_score[key] += metrics[key]

    # Average metrics over all batches
    for key in metrics_score:
        metrics_score[key] /= len(dataloader)

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken
    print(f"Mean FPS: {mean_fps:.4f}")
    print("Metrics on Test Set:")
    for key, value in metrics_score.items():
        print(f"{key.capitalize()}: {value:.4f}")

    return metrics_score

if __name__ == "__main__":
    # Seeding for reproducibility
    seeding(42)

    # Load task configuration
    with open('tasks_config.yaml', 'r') as f:
        task_configs = yaml.safe_load(f)
    task = 'Task01_BrainTumour'
    config = task_configs[task]
    modalities = config['modalities']
    in_channels = config['in_channels']
    num_classes = config['num_classes']
    slice_axis = config['slice_axis']

    # Dataset paths (Adjust these paths as necessary for your data)
    image_dir = f'./dataset/{task}/imagesTr'
    mask_dir = f'./dataset/{task}/labelsTr'
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.nii.gz')))

    # Create test dataset and dataloader
    test_dataset = MSDDataset(image_paths, mask_paths, modalities=modalities, slice_axis=slice_axis)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Load the model and weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_doubleunet(in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)

    # Load the best model checkpoint
    checkpoint_path = f"models/{task}_best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Evaluate on test data
    metrics_score = evaluate(model, test_loader, device)

    # Optionally save metrics
    # results_path = "./results/test_metrics.pkl"
    # create_dir(os.path.dirname(results_path))
    # with open(results_path, 'wb') as f:
    #     pickle.dump(metrics_score, f)

    print("Evaluation complete. Metrics saved.")
