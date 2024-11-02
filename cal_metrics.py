import pickle
import numpy as np

# Load metrics from the pickle file
with open('./results/metrics_history.pkl', 'rb') as f:
    metrics_history = pickle.load(f)

# Function to calculate overall metrics
def calculate_overall_metrics(metrics_history):
    overall_metrics = {}

    # Iterate over each phase (e.g., 'train', 'val')
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

# Calculate the overall metrics
overall_metrics = calculate_overall_metrics(metrics_history)

# Display the results
for phase in overall_metrics:
    print(f"\n{phase.upper()} METRICS:")
    for metric, stats in overall_metrics[phase].items():
        print(f"{metric}: Mean = {stats['mean']:.4f}, Std Dev = {stats['std_dev']:.4f}")
