import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from metrics import precision, recall, F2, dice_score, jac_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Shuffle the dataset. """
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

def otsu_mask(image, size):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = th.astype(np.int32)
    th = th/255.0
    th = th > 0.5
    th = th.astype(np.int32)
    return th

def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]

def plot_metrics(metrics_history, metric_names, save_path=None):
    epochs = range(1, len(metrics_history['train'][metric_names[0]]) + 1)

    num_metrics = len(metric_names)
    plt.figure(figsize=(5 * num_metrics, 5))
    for idx, metric_name in enumerate(metric_names):
        plt.subplot(1, num_metrics, idx + 1)
        plt.title(f'{metric_name} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)

        # Training metric
        train_metric_mean = [x[0] for x in metrics_history['train'][metric_name]]
        train_metric_std = [x[1] for x in metrics_history['train'][metric_name]]
        plt.plot(epochs, train_metric_mean, label=f'Training {metric_name}')
        plt.fill_between(
            epochs,
            np.array(train_metric_mean) - np.array(train_metric_std),
            np.array(train_metric_mean) + np.array(train_metric_std),
            alpha=0.3
        )

        # Validation metric
        val_metric_mean = [x[0] for x in metrics_history['val'][metric_name]]
        val_metric_std = [x[1] for x in metrics_history['val'][metric_name]]
        plt.plot(epochs, val_metric_mean, label=f'Validation {metric_name}')
        plt.fill_between(
            epochs,
            np.array(val_metric_mean) - np.array(val_metric_std),
            np.array(val_metric_mean) + np.array(val_metric_std),
            alpha=0.3
        )

        plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()