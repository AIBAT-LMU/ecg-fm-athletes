import torch
import mlflow
from loguru import logger
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler
from ecgath.metrics.metrics import MetricsManager
from ecgath.utils.logging_utils import  mlflow_log_metric_safe

def train_loop(
    model: nn.Module,
    trainloader: DataLoader,
    optimizer: Optimizer,
    lr_scheduler: Optional[_LRScheduler],
    scaler: GradScaler,
    loss_fn: nn.Module,
    clip_norm: float,
    metrics_manager: MetricsManager,
    device: torch.device,
    log_every: int = 100,
    epoch: int = 0
) -> None:
    """
    Training loop for one epoch.

    Args:
        model (nn.Module): The neural network model to train.
        trainloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        lr_scheduler (_LRScheduler, optional): Learning rate scheduler.
        scaler (GradScaler): Gradient scaler for mixed-precision training.
        loss_fn (nn.Module): Loss function.
        clip_norm (float): Maximum norm for gradient clipping (0 disables clipping).
        metrics_manager (MetricsManager): Object to track and compute training metrics.
        device (torch.device): Device to run training on (e.g., "cuda" or "cpu").
        log_every (int, optional): Interval (in batches) for logging. Defaults to 100.

    Returns:
        None
    """
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)

    for batch_idx, data in enumerate(trainloader):
        global_step = epoch * num_batches + batch_idx

        X = data.get("signals").to(device)
        y = data.get("labels").to(device)
        attention_mask = data.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        with torch.autocast(device.type):
            logits = (
                model(X, attention_mask=attention_mask)["out"]
                if attention_mask is not None
                else model(X)["out"]
            )
            loss = loss_fn(logits, y.float())

        scaler.scale(loss).backward()

        if clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler and not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        metrics_manager.update(probs, y.long())

        if batch_idx % log_every == 0:
            mlflow_log_metric_safe("batch_loss", loss.item(), step=global_step)
            logger.info(f"Batch {batch_idx}/{num_batches} - Loss: {loss.item():.6f}")

    avg_loss = total_loss / num_batches
    metrics = metrics_manager.compute()
    metrics_manager.reset()
    mlflow_log_metric_safe("avg_train_loss", avg_loss, step=global_step)
    logger.info(f"Epoch Results: Avg loss: {avg_loss:.6f}")
    mlflow_log_metric_safe(f"train_auroc", metrics['auroc'].mean(), step=global_step) # # pot. use: torch.nanmean 
    mlflow_log_metric_safe(f"train_f1", metrics['f1-score'].mean(), step=global_step)
    mlflow_log_metric_safe(f"train_auprc", metrics['auprc'].mean(), step=global_step)
    mlflow_log_metric_safe(f"train_recall", metrics['recall'].mean(), step=global_step)
    mlflow_log_metric_safe(f"train_precision", metrics['precision'].mean(), step=global_step)
    mlflow_log_metric_safe(f"train_specificity", metrics['specificity'].mean(), step=global_step)


def eval_loop(
    dataloader: DataLoader,
    model: nn.Module,
    metrics_manager: MetricsManager,
    loss_fn: nn.Module,
    device: torch.device = torch.device("cuda"),
    num_labels: Optional[int] = None,
    epoch: int = 0
) -> Dict[str, Any]:
    """
    Evaluation loop for a multilabel classification model.

    Args:
        dataloader (DataLoader): DataLoader for evaluation/validation data.
        model (nn.Module): The neural network model to evaluate.
        metrics_manager (MetricsManager): Object to track and compute evaluation metrics.
        loss_fn (nn.Module): Loss function.
        device (torch.device, optional): Device to run evaluation on. Defaults to "cuda".
        num_labels (int, optional): Number of labels (not directly used, for compatibility).

    Returns:
        Dict[str, Any]: Dictionary of computed evaluation metrics.
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    logger.info("Starting evaluation loop")
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader, 1):
            X = data.get("signals").to(device)
            y = data.get("labels").to(device)
            attention_mask = data.get("attention_mask", None)

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                logits = model(X, attention_mask=attention_mask)["out"]
            else:
                logits = model(X)["out"]

            loss = loss_fn(logits, y.float())
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            metrics_manager.update(probs, y.long())

    avg_loss = total_loss / num_batches
    metrics = metrics_manager.compute()
    metrics_manager.reset()
   
    eval_step = epoch * num_batches
    mlflow_log_metric_safe("avg_val_loss", avg_loss, step=eval_step)
    logger.info(f"Eval Results: Avg loss: {avg_loss:.6f}")
    mlflow_log_metric_safe(f"val_auroc", metrics['auroc'].mean(), step=eval_step) # pot. use: torch.nanmean 
    mlflow_log_metric_safe(f"val_f1", metrics['f1-score'].mean(), step=eval_step)
    mlflow_log_metric_safe(f"val_auprc", metrics['auprc'].mean(), step=eval_step)
    mlflow_log_metric_safe(f"val_recall", metrics['recall'].mean(), step=eval_step)
    mlflow_log_metric_safe(f"val_precision", metrics['precision'].mean(), step=eval_step)
    mlflow_log_metric_safe(f"val_specificity", metrics['specificity'].mean(), step=eval_step)
    return {"avg_loss": avg_loss, **metrics}
