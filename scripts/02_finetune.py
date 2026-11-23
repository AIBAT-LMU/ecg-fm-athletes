# imports
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import pandas as pd
from loguru import logger
from math import ceil
from typing import Any, Dict
from argparse import Namespace
from transformers import get_linear_schedule_with_warmup
import mlflow
import os
import numpy as np
import argparse
import yaml

# project specific imports
from ecgath import models
from ecgath.metrics.metrics import MetricsManager
from ecgath.data.dataloaders import create_dataloader
from ecgath.training.train_eval_loops import train_loop, eval_loop


# model builders
MODEL_BUILDERS = {
    "hubertecgsmall": lambda store, num_labels: models.hubert_ecg.build_classifier(model_store=store, size="small", num_labels=num_labels),
    "hubertecgbase": lambda store, num_labels: models.hubert_ecg.build_classifier(model_store=store, size="base", num_labels=num_labels),
    "hubertecglarge": lambda store, num_labels: models.hubert_ecg.build_classifier(model_store=store, size="large", num_labels=num_labels),
    "ecgfm": lambda store, num_labels: models.ecg_fm.build_classifier(model_store=store, num_labels=num_labels),
    "ecgfounder": lambda store, num_labels: models.ecg_founder.build_classifier(model_store=store, num_labels=num_labels),
    "ecgfounder1lead": lambda store, num_labels: models.ecg_founder.build_classifier(model_store=store, num_labels=num_labels, in_channels=1),
}

# parse_args function
def parse_args() -> Namespace:
    """Handles all config loading (YAML + command line arguments)."""
    parser = argparse.ArgumentParser(description="ECG Model Training Script")
    
    # path to config file is the only required arg
    parser.add_argument('--config', type=str, required=True, help="Path to your .yml config file.")

    # all other args are optional 
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_store', type=str)
    parser.add_argument('--model_dropout_mult', type=float)
    parser.add_argument('--transformer_blocks_to_unfreeze', type=int)
    parser.add_argument('--unfreeze_conv_embedder', type=bool)
    parser.add_argument('--split_file', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_labels', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--ramp_up_perc', type=float)
    parser.add_argument('--clip_norm', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--early_stopping_metric', type=str)
    parser.add_argument('--early_stopping_mode', type=str)
    parser.add_argument('--log_every', type=int)
    parser.add_argument('--lead', type=int)

    args = parser.parse_args()

    # load base config from the YAML (config.yml)
    with open(args.config, 'r') as f:
        config_from_yaml = yaml.safe_load(f)

    # flatten YAML sections into one dict
    flat_config = {}
    for section_values in config_from_yaml.values():
        if isinstance(section_values, dict):
            flat_config.update(section_values)
    
    # let any command line args overwrite the YAML values
    for key, value in vars(args).items():
        if value is not None:
            flat_config[key] = value

    # return as a Namespace for dot. access (e.g., args.epochs)
    return argparse.Namespace(**flat_config)


def train(args: Namespace) -> None: # now accepts args as input
    """main training and evaluation loop."""
    model_type = args.model_type.lower().replace("-", "").replace("_", "")
    mlflow.set_experiment(f"{model_type}")
    
    with mlflow.start_run():
        mlflow.log_params(vars(args))
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # find path and load class weights from label_def.csv (of each specific preprocessing)
        logger.info("Loading class weights...")
        splits_dir = os.path.dirname(args.split_file)
        label_def_path = os.path.join(splits_dir, "label_def.csv")
        logger.info(f"Loading weights from: {label_def_path}")
        label_def = pd.read_csv(label_def_path)
        pos_weights = label_def["pos_weight_train"].values
        pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float).to(device)
        logger.info(f"Loaded {len(pos_weight_tensor)} class weights.")

        # load data
        splits = pd.read_csv(args.split_file)
        trainloader = create_dataloader(
            splits, base_path=args.base_path, split="train", batch_size=args.batch_size, model_type=model_type, lead = args.lead)
        valloader = create_dataloader(
            splits, base_path=args.base_path, split="val", batch_size=args.batch_size, model_type=model_type, lead = args.lead)

        # build model
        builder = MODEL_BUILDERS.get(model_type)
        if builder is None:
            raise ValueError(f"Unknown model_type '{model_type}'")
        model = builder(args.model_store, num_labels=args.num_labels).to(device)

        # configure optimizer and scheduler based on model type
        if "hubert" in model_type:
            if hasattr(args, 'layerdrop') and args.layerdrop > 0:
                model.hubert_ecg.config.layerdrop = args.layerdrop # LayerDrop value in the model's config
                logger.info(f"Hubert-ECG: LayerDrop activated with p={args.layerdrop}")
            for name, module in model.hubert_ecg.named_modules():
                if "dropout" in name:
                    module.p = 0.1 + 0.5 * args.model_dropout_mult # "zeroing the dropout probability?" direkt auf 0?
            model.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze)
            model.set_feature_extractor_trainable(trainable=args.unfreeze_conv_embedder)
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad), lr=args.lr,
                betas=getattr(args, 'betas', (0.9, 0.98)), eps=getattr(args, 'eps', 1e-9),
                weight_decay=getattr(args, 'weight_decay_mult', 0.01) * 0.01)
            training_steps = args.epochs * len(trainloader)
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=ceil(args.ramp_up_perc * training_steps), # "warm-up for scheduler" 
                num_training_steps=training_steps) # learning rate scheduler ramps up for the first 8% of the training steps and then decays linearly to zero
        
        elif model_type == "ecgfm":
            optimizer = torch.optim.Adam(
                (p for p in model.parameters() if p.requires_grad), lr=args.lr,
                betas=getattr(args, 'betas', (0.9, 0.999)), eps=getattr(args, 'eps', 1e-8))
            # lr_scheduler = None 
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
              optimizer, 
              mode=args.early_stopping_mode, # 'max' for AUPRC
              patience=5, # reduce LR after 5 epochs without improvements of AUPRC
              factor=0.2) # reduces LR by a factor of 5 (to 20%)

        elif model_type == "ecgfounder":
            optimizer = torch.optim.Adam(
              (p for p in model.parameters() if p.requires_grad), lr=args.lr,
              weight_decay=getattr(args, 'weight_decay', 1e-5))
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
              optimizer, 
              mode=args.early_stopping_mode,
              patience=5, 
              factor=0.1)
        
        scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        metrics_manager = MetricsManager(num_labels=args.num_labels, device=device)

        # vars for early stopping
        best_metric_value = np.inf if args.early_stopping_mode == "min" else -np.inf
        epochs_no_improve = 0
        
        # checkpoint directory
        checkpoint_path = Path(args.checkpoint_path)
        # check if path is a directory
        if checkpoint_path.is_dir():
            logger.info(f"Checkpoint path is a directory. Saving model.")
            checkpoint_path = checkpoint_path / f"best_model_{model_type}.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        

        # training loop
        for epoch in range(args.epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{args.epochs}")

            train_loop(
                model=model, trainloader=trainloader, optimizer=optimizer,
                lr_scheduler=lr_scheduler, scaler=scaler, loss_fn=loss_fn,
                clip_norm=args.clip_norm, metrics_manager=metrics_manager,
                device=device, log_every=args.log_every, epoch=epoch)

            val_metrics = eval_loop(
                dataloader=valloader, model=model, metrics_manager=metrics_manager,
                loss_fn=loss_fn, device=device, num_labels=args.num_labels, epoch=epoch)
            
            current_metric_value = val_metrics[args.early_stopping_metric]
            if isinstance(current_metric_value, torch.Tensor):
                current_metric_value = current_metric_value.mean().item()
            if lr_scheduler and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(current_metric_value)

            logger.info(f"Monitored metric ({args.early_stopping_metric}): {current_metric_value:.6f}")

            
            # see if we improved
            improved = (args.early_stopping_mode == 'min' and current_metric_value < best_metric_value) or \
                       (args.early_stopping_mode == 'max' and current_metric_value > best_metric_value)

            if improved:
                # new best score, save it and reset patience
                best_metric_value = current_metric_value
                epochs_no_improve = 0
                logger.info(f"New best model. Saving checkpoint to {checkpoint_path}")
                torch.save(model.state_dict(), checkpoint_path)
            else:
                # no improvement, tick up the patience counter
                epochs_no_improve += 1
                logger.info(f"No improvement for {epochs_no_improve} epochs.")

            # stop if we've run out of patience
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping after {args.patience} epochs.")
                break
        
        logger.info("Training complete.")

        # load best model before we finish
        if checkpoint_path.exists():
            logger.info(f"Loading best model weights from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            mlflow.log_artifact(str(checkpoint_path))
        else:
            logger.warning("No checkpoint file found. Using model from last epoch.")


# script entry point
if __name__ == "__main__":
    # get all settings
    args = parse_args()
    
    # kick off  training
    train(args)
