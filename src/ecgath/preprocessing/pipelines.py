# integrates metadata/label handling and is ready for the new config management.

import pandas as pd
from argparse import Namespace

# project-specific imports
from ecgath.preprocessing.signals import process_signal
from ecgath.data.io import load_physionet, load_lmu_sportscar, load_norwegian, save_file, load_code15
from ecgath.preprocessing.metadata import (
    clean_physionet_comments,
    fetch_physionet_label_description,
    get_pyshionet_labels,
    get_lmu_sportscar_labels,
    get_norwegian_labels,
    get_code15_labels,
)

# This can be loaded once at the module level
PHYSIONET_LABEL_DESC = fetch_physionet_label_description()

def preprocess_physionet(record: pd.Series, config: Namespace) -> pd.Series:
    """
    Processes a single record from PhysioNet dataset.
    Loads data, cleans metadata, gets labels, and processes the signal.
    """
    # 1. load signal and initial metadata
    sig_array, fields = load_physionet(record["path"])

    # 2. extract and clean labels and comments (specific to PhysioNet)
    fields.update(clean_physionet_comments(fields.pop("comments", [])))
    labels, fields["unmatched_diagnosis"] = get_pyshionet_labels(fields.get("Dx", ""), PHYSIONET_LABEL_DESC)

    # 3. call the signal processing pipeline
    processed_segments, fields = process_signal(
        sig_array=sig_array,
        fields=fields,
        desired_sample_rate=config.desired_sample_rate,
        standardize_strategy=config.standardization,
        filter_strategy=config.filter_strategy,
        constant_lead_strategy=config.constant_lead_strategy,
        leads=config.leads,
        segment_len_s=config.segment_len_s,
        handle_remainder=config.handle_remainder,
        save_path=record["save_path"],
        idx=record["idx"],
        labels=labels,
    )

    # 4. save each segment to a .mat file and collect the new paths
    save_paths = []
    for i, segment_data in enumerate(processed_segments):
        # Create a unique path for each segment, e.g., 'file_0.mat', 'file_1.mat'
        segment_path = str(record["save_path"]).replace('.mat', f'_{i}.mat')
        save_file(segment_path, segment_data)
        save_paths.append(segment_path)
    
    fields['save_path'] = save_paths
    return pd.Series(fields)


def preprocess_lmu_sportscar(record: pd.Series, config: Namespace) -> pd.Series:
    """
    Processes a single record from the LMU Sportscar dataset.
    """
    # 1. load signal
    sig_array, fields = load_lmu_sportscar(record["path"])

    # 2. get labels (specific to LMU Sportscar)
    labels = get_lmu_sportscar_labels(record, PHYSIONET_LABEL_DESC)

    # 3. call the signal processing pipeline
    processed_segments, fields = process_signal(
        sig_array=sig_array,
        fields=fields,
        desired_sample_rate=config.desired_sample_rate,
        standardize_strategy=config.standardization,
        filter_strategy=config.filter_strategy,
        constant_lead_strategy=config.constant_lead_strategy,
        leads=config.leads,
        segment_len_s=config.segment_len_s,
        handle_remainder=config.handle_remainder,
        save_path=record["save_path"],
        idx=record["idx"],
        labels=labels,
    )

    # 4. save logic
    save_paths = []
    for i, segment_data in enumerate(processed_segments):
        segment_path = str(record["save_path"]).replace('.csv', f'_{i}.mat') # Note: .csv to .mat
        save_file(segment_path, segment_data)
        save_paths.append(segment_path)
        
    fields['save_path'] = save_paths
    return pd.Series(fields)


def preprocess_norwegian(record: pd.Series, config: Namespace) -> pd.Series:
    """
    Processes a single record from the Norwegian athlete dataset.
    """
    # 1. load signal
    sig_array, fields = load_norwegian(record["path"])

    # 2. extract and clean labels (specific to Norwegian dataset)
    fields.update(clean_physionet_comments(fields.pop("comments", [])))
    label_map, unmatched_diagnoses = get_norwegian_labels(fields.get("C", ""), PHYSIONET_LABEL_DESC)
    fields["unmatched_diagnosis"] = unmatched_diagnoses # Optional, but consistent with physionet

    # 3. call the generic signal processing pipeline
    processed_segments, fields = process_signal(
        sig_array=sig_array,
        fields=fields,
        desired_sample_rate=config.desired_sample_rate,
        standardize_strategy=config.standardization,
        filter_strategy=config.filter_strategy,
        constant_lead_strategy=config.constant_lead_strategy,
        leads=config.leads,
        segment_len_s=config.segment_len_s,
        handle_remainder=config.handle_remainder,
        save_path=record["save_path"],
        idx=record["idx"],
        labels=label_map,
    )

    # 4. save logic
    save_paths = []
    for i, segment_data in enumerate(processed_segments):
        segment_path = str(record["save_path"]).replace('.dat', f'_{i}.mat') # dat -> mat evtl auch für lmu
        save_file(segment_path, segment_data)
        save_paths.append(segment_path)
        
    fields['save_path'] = save_paths
    return pd.Series(fields)

def preprocess_code15(record: pd.Series, config: Namespace) -> pd.Series:
    """
    Processes a single record from the CODE-15 dataset.
    """
    # 1. load signal
    sig_array, fields = load_code15(record["path"])

    # 2. get labels
    label_map = get_code15_labels(record["metadata"], PHYSIONET_LABEL_DESC)

    # 3. call the generic signal processing pipeline
    processed_segments, fields = process_signal(
        sig_array=sig_array,
        fields=fields,
        desired_sample_rate=config.desired_sample_rate,
        standardize_strategy=config.standardization,
        filter_strategy=config.filter_strategy,
        constant_lead_strategy=config.constant_lead_strategy,
        leads=config.leads,
        segment_len_s=config.segment_len_s,
        handle_remainder=config.handle_remainder,
        save_path=record["save_path"],
        idx=record["idx"],
        labels=label_map,
    )

    # 4. save logic
    save_paths = []
    for i, segment_data in enumerate(processed_segments):
        # record["save_path"] ist hier standardmäßig schon ein .mat Pfad
        segment_path = str(record["save_path"]).replace('.mat', f'_{i}.mat')
        save_file(segment_path, segment_data)
        save_paths.append(segment_path)
        
    fields['save_path'] = save_paths
    return pd.Series(fields)