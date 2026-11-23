import pandas as pd
import numpy as np
import torch
import sys
import os
import argparse
import yaml
from argparse import Namespace
from tqdm import tqdm

# project modules
from ecgath import models 
from ecgath.data.dataloaders import create_dataloader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAUROC, MultilabelAveragePrecision, MultilabelF1Score, 
    MultilabelRecall, MultilabelPrecision, MultilabelSpecificity
)

# model builders
MODEL_BUILDERS = {
    "hubertecgsmall": lambda store, num_labels: models.hubert_ecg.build_classifier(model_store=store, size="small", num_labels=num_labels),
    "hubertecgsbase": lambda store, num_labels: models.hubert_ecg.build_classifier(model_store=store, size="base", num_labels=num_labels),
    "hubertecgslarge": lambda store, num_labels: models.hubert_ecg.build_classifier(model_store=store, size="large", num_labels=num_labels),
    "ecgfm": lambda store, num_labels: models.ecg_fm.build_classifier(model_store=store, num_labels=num_labels),
    "ecgfounder": lambda store, num_labels: models.ecg_founder.build_classifier(model_store=store, num_labels=num_labels),
}

def generate_predictions(model, args, device, label_cols):
    """
    Step 1: Generates segment-level predictions for the 'test' split.
    """
    print("\nStarting segment-level predictions...")
    
    dataloader = create_dataloader(
        pd.read_csv(args.split_file), 
        base_path=args.base_path, 
        split='test', 
        batch_size=args.batch_size, 
        model_type=args.model_type.lower().replace("-", "").replace("_", "")
    )
    
    if dataloader is None:
        print("ERROR: Test split not found or is empty. Cannot generate predictions.")
        sys.exit(1)

    all_probs = []
    all_segment_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting on test set"):
            X = batch["signals"].to(device)
            segment_ids = batch["ecg_id"] 
            attention_mask = batch.get("attention_mask", None)
            
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                logits = model(X, attention_mask=attention_mask)["out"]
            else:
                logits = model(X)["out"]
            
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_segment_ids.extend(segment_ids)

    all_probs_tensor = torch.cat(all_probs, dim=0)
    class_names = [c.replace('label_', '') for c in label_cols]
    
    pred_df = pd.DataFrame(all_probs_tensor.numpy(), columns=class_names)
    pred_df.index = all_segment_ids
    pred_df.index.name = 'ecg_segment_id'

    raw_preds_output_path = os.path.join(args.output_dir, "raw_segment_predictions.csv")
    pred_df.to_csv(raw_preds_output_path)

    print(f"\nRaw segment predictions saved to: {raw_preds_output_path}")
    print("\n Segment predictions generated.")
    return pred_df

def analyze_aggregated_results(pred_segments_df, true_athlete_df, output_filename):
    """
    Step 2: Aggregates predictions and computes athlete-level metrics.
    """
    # 1. Aggregation & Alignment
    patient_ids = [f.split('_')[0] if isinstance(f, str) and '_' in f else str(f) for f in pred_segments_df.index.values]
    pred_segments_df['id'] = patient_ids
    pred_athlete_probs = pred_segments_df.groupby('id').mean()

    # The column merges are specific to the LMU/PhysioNet label mapping
    column_merges = {'CLBBB|LBBB': ['CLBBB', 'LBBB'], 'CRBBB|RBBB': ['CRBBB', 'RBBB'], 
                     'PAC|SVPB': ['PAC', 'SVPB'], 'PVC|VPB': ['PVC', 'VPB']}
    for new_col, old_cols in column_merges.items():
        if all(col in true_athlete_df.columns for col in old_cols):
            true_athlete_df[new_col] = true_athlete_df[old_cols].any(axis=1).astype(int)
            true_athlete_df = true_athlete_df.drop(columns=old_cols, errors='ignore')

    common_ids = sorted(set(true_athlete_df.index.astype(str)) & set(pred_athlete_probs.index.astype(str)))
    if len(common_ids) < len(true_athlete_df):
        missing = set(true_athlete_df.index.astype(str)) - set(pred_athlete_probs.index.astype(str))
        print(f"⚠️ Warning: Ignoring {len(missing)} athletes without ECG data.")

    true_athlete_final = true_athlete_df.loc[common_ids].sort_index()
    pred_athlete_final_probs = pred_athlete_probs.loc[common_ids].sort_index()

    common_labels = sorted(list(set(true_athlete_final.columns) & set(pred_athlete_final_probs.columns)))
    true_athlete_final = true_athlete_final[common_labels]
    pred_athlete_final_probs = pred_athlete_final_probs[common_labels]

    print(f" Analysis on {len(common_ids)} athletes and {len(common_labels)} common labels.")

    # 2. Metric Calculation
    preds_tensor = torch.tensor(pred_athlete_final_probs.values, dtype=torch.float32)
    labels_tensor = torch.tensor(true_athlete_final.values, dtype=torch.int64) 
    num_labels = len(common_labels)
    metrics_collection = MetricCollection({
        'AUROC': MultilabelAUROC(num_labels=num_labels, average='none'), 
        'AUPRC': MultilabelAveragePrecision(num_labels=num_labels, average='none'),
        'F1-Score': MultilabelF1Score(num_labels=num_labels, average='none'), 
        'Recall': MultilabelRecall(num_labels=num_labels, average='none'), 
        'Precision': MultilabelPrecision(num_labels=num_labels, average='none'),
        'Specificity': MultilabelSpecificity(num_labels=num_labels, average='none')
    })
    results_per_class = metrics_collection(preds_tensor, labels_tensor)

    # 3. Report Generation
    report_df = pd.DataFrame({'Class': common_labels})
    support_tensor = labels_tensor.sum(axis=0)
    n_total_tensor = torch.tensor(float(len(common_ids)))
    valid_mask = (support_tensor > 0) & (support_tensor < n_total_tensor)

    for metric_name in results_per_class:
        raw_metric_tensor = results_per_class[metric_name].clone()
        if metric_name in ['AUROC', 'AUPRC', 'F1-Score']:
            raw_metric_tensor[~valid_mask] = np.nan
        if metric_name == 'Specificity':
            raw_metric_tensor[support_tensor == n_total_tensor] = np.nan
        if metric_name in ['Recall', 'Precision', 'F1-Score']:
            raw_metric_tensor[support_tensor == 0] = np.nan
        report_df[metric_name] = raw_metric_tensor.numpy()

    report_df['Support'] = support_tensor.numpy()
    
    true_global_metrics = {'Class': 'GLOBAL (Macro-Avg)'}
    for col in report_df.columns[1:-1]:
        true_global_metrics[col] = np.nanmean(report_df[col].values).item()
    report_df.loc[len(report_df)] = true_global_metrics

    print("\n--- RESULTS ---")
    print(report_df.to_markdown(index=False, floatfmt=".4f"))
    report_df.to_csv(output_filename, index=False)
    print(f"\nReport saved to '{output_filename}'")

def main(cli_args):
    with open(cli_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    args = Namespace(**{k: v for section in config_dict.values() for k, v in section.items()})
    args.output_dir = cli_args.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = MODEL_BUILDERS.get(args.model_type.lower().replace("-", "").replace("_", ""))
    
    model = builder(args.model_store, num_labels=args.num_labels).to(device)
    model.load_state_dict(torch.load(cli_args.checkpoint, map_location=device))
    model.eval()

    splits_df = pd.read_csv(args.split_file)
    label_cols = [c for c in splits_df.columns if c.startswith('label_')]

    output_dir = cli_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    pred_segments_df = generate_predictions(model, args, device, label_cols)

    true_athlete_df = pd.read_csv(args.base_path + "/data/raw/lmu_sportscar/ECG_Data_Labeled.csv", index_col=0).replace({'Ja': 1, 'Nein': 0})

    output_path = os.path.join(cli_args.output_dir, "athlete_aggregated_metrics_report.csv")
    
    analyze_aggregated_results(pred_segments_df, true_athlete_df, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Evaluation of ECG Models on Athlete Data")
    parser.add_argument('--config', required=True, help="Path to the YAML config file.")
    parser.add_argument('--checkpoint', required=True, help="Path to the model checkpoint.")
    parser.add_argument('--output_dir', required=True, help="Directory to save the final report.")
    cli_args = parser.parse_args()
    main(cli_args)