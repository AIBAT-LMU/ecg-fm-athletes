import mlflow
import torch
import numpy as np

def mlflow_log_metric_safe(key, value, step=None):
    if not mlflow.active_run():
        return

    try:
        # Handle torch tensors
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                mlflow.log_metric(key, value.item(), step=step)
            else:
                for i, v in enumerate(value):
                    mlflow.log_metric(f"{key}_label_{i}", v.item(), step=step)

        # Handle numpy arrays
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                mlflow.log_metric(key, float(value), step=step)
            else:
                for i, v in enumerate(value):
                    mlflow.log_metric(f"{key}_label_{i}", float(v), step=step)

        # Handle lists or tuples
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                mlflow.log_metric(f"{key}_label_{i}", float(v), step=step)

        # Handle scalar values
        else:
            mlflow.log_metric(key, float(value), step=step)

    except Exception as e:
        print(f"[Warning] Failed to log metric '{key}': {e}")
