import torch
import pandas as pd
import yaml
import os
import argparse
from tqdm import tqdm
from argparse import Namespace

# project modules
from ecgath import models
from ecgath.data.dataloaders import create_dataloader
from ecgath.metrics.metrics import MetricsManager

# model builders
MODEL_BUILDERS = {
    "hubertecgsmall": lambda store, num_labels: models.hubert_ecg.build_classifier(model_store=store, size="small", num_labels=num_labels),
    "hubertecgsbase": lambda store, num_labels: models.hubert_ecg.build_classifier(model_store=store, size="base", num_labels=num_labels),
    "hubertecgslarge": lambda store, num_labels: models.hubert_ecg.build_classifier(model_store=store, size="large", num_labels=num_labels),
    "ecgfm": lambda store, num_labels: models.ecg_fm.build_classifier(model_store=store, num_labels=num_labels),
    "ecgfounder": lambda store, num_labels: models.ecg_founder.build_classifier(model_store=store, num_labels=num_labels),
}

def analyze_split(model, args, split_name, device, label_cols):
    print(f"\nStarting evaluation for split {split_name.upper()}")
    
    dataloader = create_dataloader(
        pd.read_csv(args.split_file), 
        base_path=args.base_path, 
        split=split_name, 
        batch_size=args.batch_size, 
        model_type=args.model_type.lower().replace("-", "").replace("_", "")
    )
    
    if dataloader is None: # for evaluation where we only use the "test split"
        print(f"Skipping split {split_name.upper()}: No data found in splits.csv.")
        return pd.DataFrame() # return empty df
    metrics_manager = MetricsManager(num_labels=args.num_labels, device=device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting on {split_name}"):
            X = batch["signals"].to(device)
            y = batch["labels"]
            
            # check for attention mask (only for hubert)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                logits = model(X, attention_mask=attention_mask)["out"]
            else:
                logits = model(X)["out"]

            preds = torch.sigmoid(logits)
            metrics_manager.update(preds, y.long())

    results_per_class = metrics_manager.compute()
    
    class_names = [c.replace('label_', '') for c in label_cols]
    report_df = pd.DataFrame({
        'Class': class_names,
        'AUPRC': results_per_class['auprc'].numpy(),
        'AUROC': results_per_class['auroc'].numpy(),
        'F1-Score': results_per_class['f1-score'].numpy(),
        'Recall': results_per_class['recall'].numpy(),
        'Precision': results_per_class['precision'].numpy(),
        'Specificity': results_per_class['specificity'].numpy(),
    })

    # macro averages for all metrics
    global_metrics = {
        'Class': 'GLOBAL',
        'AUPRC': torch.nanmean(results_per_class['auprc']).item(),
        'AUROC': torch.nanmean(results_per_class['auroc']).item(),
        'F1-Score': torch.nanmean(results_per_class['f1-score']).item(),
        'Recall': torch.nanmean(results_per_class['recall']).item(),
        'Precision': torch.nanmean(results_per_class['precision']).item(),
        'Specificity': torch.nanmean(results_per_class['specificity']).item(),
    }

    report_df.loc[len(report_df)] = global_metrics

    print(f"Global Macro-AUPRC: {global_metrics['AUPRC']:.4f}")
    print(f"Global Macro-AUROC: {global_metrics['AUROC']:.4f}")
    print(f"Global Macro-F1: {global_metrics['F1-Score']:.4f}")
    print(f"Global Macro-Recall: {global_metrics['Recall']:.4f}")
    print(f"Global Macro-Precision: {global_metrics['Precision']:.4f}")
    print(f"Global Macro-Specificity: {global_metrics['Specificity']:.4f}")
    
    return report_df

def analyze_all_splits(model, args, label_cols, device, output_dir):
    global_rows = []

    for split in ['train', 'val', 'test']:
        df = analyze_split(model, args, split, device, label_cols)

        if df.empty: 
            continue # skip if there is no report

        output_path = os.path.join(output_dir, f"evaluation_results_{split}.csv")
        df.to_csv(output_path, index=False)
        print(f"Detailed results for {split} saved to {output_path}")

        global_row = df[df['Class'] == 'GLOBAL'].copy()
        global_row['Split'] = split.upper()
        global_rows.append(global_row)

    global_summary_df = pd.concat(global_rows, ignore_index=True)
    global_summary_df = global_summary_df[['Split', 'AUPRC', 'AUROC', 'F1-Score', 'Recall', 'Precision', 'Specificity']]
    summary_path = os.path.join(output_dir, "GLOBAL_summary.csv")
    global_summary_df.to_csv(summary_path, index=False)
    print(f"GLOBAL summary saved to {summary_path}")

def main(cli_args):
    with open(cli_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    flat_config = {}
    for section in config_dict.values():
        if isinstance(section, dict):
            flat_config.update(section)
    args = Namespace(**flat_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    builder = MODEL_BUILDERS.get(args.model_type.lower().replace("-", "").replace("_", ""))
    model = builder(args.model_store, num_labels=args.num_labels).to(device)
    model.load_state_dict(torch.load(cli_args.checkpoint, map_location=device))
    model.eval()

    splits_df = pd.read_csv(args.split_file)
    label_cols = [c for c in splits_df.columns if c.startswith('label_')]

    output_dir = cli_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    analyze_all_splits(model, args, label_cols, device, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script for ECG Models")
    parser.add_argument('--config', required=True, help="Path to YAML config")
    parser.add_argument('--checkpoint', required=True, help="Path to model checkpoint")
    parser.add_argument('--output_dir', required=True, help="Directory for CSV results")
    cli_args = parser.parse_args()
    main(cli_args)

