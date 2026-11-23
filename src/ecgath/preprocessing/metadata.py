import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from thefuzz import process


def clean_physionet_comments(comment_list: List[str]) -> Dict[str, str]:
    """
    Converts a list of colon-separated comment strings into a dictionary.

    Args:
        comment_list (List[str]): List of strings like ["Key: Value", ...]

    Returns:
        Dict[str, str]: Parsed key-value pairs from the list
    """
    return {
        item.split(":", 1)[0].strip(): item.split(":", 1)[1].strip()
        for item in comment_list
        if ":" in item
    }


def fetch_physionet_label_description() -> pd.DataFrame:
    """
    Loads and combines PhysioNet 2021 label abbreviations, full descriptions,
    and SNOMED CT codes into a unified DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns [Label, Description, SNOMED CT code]
    """
    url_abbrev = "https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/weights_abbreviations.csv"
    url_long_desc = "https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_scored.csv"
    url_snomed = (
        "https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/weights.csv"
    )

    label_abbrevs = pd.read_csv(url_abbrev, index_col=0).index.to_list()
    long_desc_df = pd.read_csv(url_long_desc, index_col=0)
    snomed_codes = pd.read_csv(url_snomed, index_col=0).index.to_list()

    abbrev_to_desc = {row.Abbreviation: row.name for _, row in long_desc_df.iterrows()}

    def build_label_description(abbrevs: List[str], mapping: Dict[str, str]) -> Dict[str, str]:
        updated = {}
        for label in abbrevs:
            parts = label.split("|")
            descriptions = [mapping.get(p, p) for p in parts]
            updated[label] = ", ".join(descriptions)
        return updated

    descriptions = build_label_description(label_abbrevs, abbrev_to_desc)

    return pd.DataFrame(
        {
            "Label": label_abbrevs,
            "Description": [descriptions[label] for label in label_abbrevs],
            "SNOMED CT code": snomed_codes,
        }
    )


def fuzzy_map(diagnosis: str, choices: List[str], threshold: int = 85) -> Optional[str]:
    """
    Maps a diagnosis string to the closest match in a list of choices using fuzzy string matching.

    Args:
        diagnosis (str): The diagnosis string to match.
        choices (List[str]): A list of possible correct labels (e.g., from a DataFrame).
        threshold (int, optional): Minimum similarity score (0–100) required to accept a match. Defaults to 90.

    Returns:
        Optional[str]: The best matching label if the score exceeds the threshold, otherwise None.
    """
    diagnosis = diagnosis.lower()
    match, score = process.extractOne(diagnosis, choices)
    if score >= threshold:
        return match
    return None


def get_pyshionet_labels(
    dx_string: str, label_description_df: pd.DataFrame
) -> Tuple[Dict[str, int], List[str]]:
    """
    Maps raw diagnosis string to PhysioNet labels and identifies unmatched SNOMED codes.

    Args:
        dx_string (str): Raw diagnosis string, e.g., '426783006, 429622005'
        label_description_df (pd.DataFrame): Output of `fetch_physionet_label_description`

    Returns:
        Tuple:
            Dict[str, int]: {Label: 0 or 1}, indicating presence in dx_string
            List[str]: SNOMED codes from dx_string not matched to any label
    """
    dx_codes = [code.strip() for code in dx_string.split(",") if code.strip()]
    all_known_snomed = set()

    for snomed_code_str in label_description_df["SNOMED CT code"]:
        all_known_snomed.update(snomed_code_str.split("|"))

    label_map = {}
    for _, row in label_description_df.iterrows():
        snomed_codes = row["SNOMED CT code"].split("|")
        label_map[row["Label"]] = int(any(code in snomed_codes for code in dx_codes))

    unmatched = [code for code in dx_codes if code not in all_known_snomed]
    return label_map, unmatched


def get_lmu_sportscar_labels(
    record: pd.Series, label_description_df: pd.DataFrame
) -> Dict[str, int]:
    """
    Extracts one-hot encoded binary labels for a single record in the LMU SportsCar dataset.

    Args:
        record (pd.Series): A single patient record containing binary labels.
        label_description_df (pd.DataFrame): DataFrame containing a 'Label' column with label names.

    Returns:
        Dict[str, int]: Dictionary mapping label names to 0 or 1, indicating absence or presence.
    """
    label_map = {label: int(record[label]) for label in label_description_df.Label}

    return label_map


def get_norwegian_labels(
    c_string: str, label_description_df: pd.DataFrame
) -> Tuple[Dict[str, int], List[str]]:
    """
    Maps diagnoses from a Norwegian athlete record string to one-hot encoded binary labels using fuzzy matching.

    Args:
        c_string (str): Comma-separated diagnosis string (e.g. "normal sinus rhythm, left axis deviation").
        label_description_df (pd.DataFrame): DataFrame with 'Description' and 'Label' columns.

    Returns:
        Tuple[Dict[str, int], List[str]]:
            - Dictionary of label names mapped to 0 or 1 (one-hot encoding).
            - List of diagnosis strings from `c_string` that could not be matched.
    """
    diagnoses = [diag.strip().lower() for diag in c_string.split(",") if diag.strip()]
    descriptions = label_description_df["Description"].tolist()

    label_map = {}
    unmatched = []
    matches = []
    for diag in diagnoses:
        match = fuzzy_map(diag, descriptions)
        if match:
            matches.append(match)
        else:
            unmatched.append(diag)
        #
    for _, row in label_description_df.iterrows():
        label_map[row["Label"]] = int(any(match in row.Description for match in matches))

    return label_map, unmatched


def get_code15_labels(metadata: dict, label_description_df: pd.DataFrame) -> Dict[str, int]:
    label_map = {row["Label"]: 0 for _, row in label_description_df.iterrows()}
    
    # Prüft auf True (Boolean), 1 (Integer) oder "True" (String)
    if metadata.get("AF") in [True, 1, "True"]: label_map["AF"] = 1
    if metadata.get("SB") in [True, 1, "True"]: label_map["SB"] = 1
    if metadata.get("ST") in [True, 1, "True"]: label_map["STach"] = 1
    if metadata.get("1dAVb") in [True, 1, "True"]: label_map["IAVB"] = 1
    if metadata.get("RBBB") in [True, 1, "True"]: label_map["CRBBB|RBBB"] = 1
    if metadata.get("LBBB") in [True, 1, "True"]: label_map["CLBBB|LBBB"] = 1
        
    return label_map