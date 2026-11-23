# Description: Unified script for preprocessing various ECG datasets.
# reads all settings from a YAML config file.

import argparse
import logging
from pathlib import Path
import pandas as pd
import yaml
from argparse import Namespace
from importlib.util import find_spec
import os
import inspect # import for smart parameter filtering

# project specific imports
from ecgath.preprocessing.pipelines import preprocess_physionet, preprocess_lmu_sportscar, preprocess_norwegian, preprocess_code15
from ecgath.preprocessing.records import collect_physionet_files, collect_lmu_sportscar_files, collect_norwegian_files, collect_code15_files
from ecgath.preprocessing.splitting import split_stratified, split_random, get_label_weights


# dispatcher dicts
SOURCE_LOADERS = {
    "physionet2021": collect_physionet_files,
    "lmu_sportscar": collect_lmu_sportscar_files,
    "norwegian_athletes": collect_norwegian_files,
    "code15": collect_code15_files,
}

PREPROCESSORS = {
    "physionet2021": preprocess_physionet,
    "lmu_sportscar": preprocess_lmu_sportscar,
    "norwegian_athletes": preprocess_norwegian,
    "code15": preprocess_code15,
}

SPLITTERS = {
    "stratified": split_stratified,
    "random": split_random,
}


###### HELPER FUNCTIONS ######
def setup_parallel_processing(no_parallel, nb_workers):
    """Initializes pandas for parallel or sequential processing."""
    if no_parallel:
        from tqdm import tqdm
        tqdm.pandas()
        return "progress_apply"
    else:
        if find_spec("pandarallel") is None:
            raise ImportError("Please install `pandarallel` or use --no_parallel.")
        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=True, nb_workers=nb_workers or os.cpu_count())
        return "parallel_apply"

def expand_array_column(df, colname):
    """Expands a DataFrame column containing lists/arrays into separate columns."""
    expanded = df[colname].apply(pd.Series)
    expanded.columns = [f"{colname}_{i}" for i in expanded.columns]
    return expanded


###### CONFIGURATION LOGIC ######
def parse_args() -> Namespace:
    """Loads configuration from a YAML file."""
    parser = argparse.ArgumentParser(description="Unified ECG Preprocessing Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the preprocessing_config.yml file.")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # convert dictionary to namespace for easy access (e.g., config.io.raw_root)
    config = Namespace(**{k: Namespace(**v) if isinstance(v, dict) else v for k, v in config_dict.items()})
    
    return config


######## MAIN LOGIC #########
def main(config: Namespace):
    """Main function to run the entire preprocessing pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting preprocessing...")
    
    # 1. setup paths and source
    raw_root = Path(config.io.raw_root)
    processed_root = Path(config.io.processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)
    source = config.io.source

    if source not in SOURCE_LOADERS:
        raise ValueError(f"Unknown source: {source}. Available: {list(SOURCE_LOADERS.keys())}")

    # 2. collect records
    logging.info(f"Collecting records for source: {source}")
    records = SOURCE_LOADERS[source](raw_root, source)
    records.index.name = "idx"
    records.reset_index(inplace=True)
    
    # 3. exclude specified datasets based on config
    if hasattr(config.io, 'exclude_datasets') and config.io.exclude_datasets:
        original_rows = len(records)
        datasets_to_exclude = config.io.exclude_datasets
        logging.info(f"Excluding datasets: {datasets_to_exclude}")
        records = records[~records['dataset'].isin(datasets_to_exclude)].copy()
        logging.info(f"Removed {original_rows - len(records)} rows belonging to excluded datasets.")

    # THIS LINE IS FOR TESTING ONLY! comment out if needed
    # records = records.head(2500)
    
    records.loc[:, "path"] = records["path"].apply(lambda x: raw_root / x)
    records.loc[:, "source"] = source
    records.loc[:, "save_path"] = records["save_file"].apply(lambda x: processed_root / x)

    # 4. run preprocessing pipeline
    logging.info("Processing signals...")
    meta = records.copy()
    apply_method = setup_parallel_processing(config.execution.no_parallel, config.execution.nb_workers)
    apply_fn = getattr(records, apply_method)
    
    processed_fields = apply_fn(
        lambda row: PREPROCESSORS[source](record=row, config=config.pipeline_params),
        axis=1
    )

    # 5. post-process and save metadata
    logging.info("Post-processing and saving metadata...")
    if 'save_path' in meta.columns:
        meta = meta.drop(columns=['save_path'])
    meta = pd.concat([meta, processed_fields], axis=1)

    array_cols = [col for col in ["mean", "std", "save_path"] if col in meta.columns]
    for col in array_cols:
        if meta[col].apply(lambda x: isinstance(x, list)).any():
            expanded = expand_array_column(meta, col)
            meta = pd.concat([meta.drop(columns=[col]), expanded], axis=1)
    
    meta_path = processed_root / f"{source}_meta.csv"
    meta.to_csv(meta_path, index=False)
    logging.info(f"Metadata saved to {meta_path}")

    # 6. filter bad data
    logging.info("Filtering out bad data based on config...")
    if hasattr(config.pipeline_params, 'filter_cols') and config.pipeline_params.filter_cols:
        original_rows = len(meta)
        filter_cols = config.pipeline_params.filter_cols
        
        # do all filter columns exist in the dataframe before trying to filter?
        existing_filter_cols = [col for col in filter_cols if col in meta.columns]
        
        if not existing_filter_cols:
            logging.warning("None of the specified filter_cols found in metadata. Skipping filtering.")
        else:
            logging.info(f"Applying filters for columns: {existing_filter_cols}")
            # keep only rows where NONE of the error flags are True
            meta = meta[~meta[existing_filter_cols].any(axis=1)].copy()
            logging.info(f"Removed {original_rows - len(meta)} rows based on filters.")
    else:
        logging.info("No filter_cols specified in config. Skipping filtering.")

    # 7. split data 
    logging.info(f"Splitting data using '{config.splitting.strategy}' strategy...")
    splitter_fn = SPLITTERS[config.splitting.strategy]
    all_possible_params = getattr(config.splitting, 'params', {}) or {}

    func_signature = inspect.signature(splitter_fn)
    accepted_params = func_signature.parameters.keys()
    filtered_params = {k: v for k, v in all_possible_params.items() if k in accepted_params}

    logging.info(f"Calling {splitter_fn.__name__} with parameters: {filtered_params}")
    
    splitted = splitter_fn(meta, fractions=config.splitting.fractions, **filtered_params)
    splits_path = processed_root / 'splits.csv'
    splitted.to_csv(splits_path, index=False)
    logging.info(f"Splits saved to {splits_path}")

    # 8. calculate and save label weights
    logging.info("Calculating and saving label weights...")
    label_def = get_label_weights(splitted)
    label_def_path = processed_root / 'label_def.csv'
    label_def.to_csv(label_def_path)
    logging.info(f"Label definitions saved to {label_def_path}")
    
    if 'pos_weight_train' in label_def.columns:
        pos_weight_str = '[' + ','.join(
            (label_def['pos_weight_train']).round(decimals=6).astype(str).values
        ) + ']'
        with open(str(processed_root / 'pos_weight.txt'), 'w') as f:
            f.write(pos_weight_str)

    logging.info("Preprocessing complete.")


if __name__ == "__main__":
    config = parse_args()
    main(config)
