import glob
import os
from pathlib import Path
from typing import Union
import pandas as pd
from ecgath.utils.files import remove_common_segments, remove_ext


def collect_physionet_files(raw_root: Union[str, Path], source: str) -> pd.DataFrame:
    """
    Recursively collects all .mat or .dat files from a PhysioNet-style dataset directory,
    removes shared path prefixes, and extracts the dataset name from the top-level directory.

    Args:
        raw_root (Union[str, Path]): The root directory containing the PhysioNet dataset files.
        source (str): Dataset source identifier (e.g., 'physionet2021') used to determine the file extension.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'path': Relative path to each file (prefix removed).
            - 'dataset': Name of the top-level dataset directory each file belongs to.
            - 'save_file': Constructed save filename based on path and source.
    """
    raw_root = Path(raw_root)  # Convert if it's a string
    if source == "physionet2021":
        ext = ".mat"
    else:
        ext = ".dat"

    pattern = str(raw_root / "**" / f"*{ext}")
    files = remove_ext(glob.glob(pattern, recursive=True))

    matched_files = [f for f in files if os.path.isfile(f + ".hea")]
    relative_paths = remove_common_segments(matched_files)
    records = pd.DataFrame({"path": relative_paths})
    records["dataset"] = records["path"].str.extract(r"^([^\\/]+)", expand=False)

    if "save_file" not in records.columns:
        records.loc[:, "save_file"] = (
            records["path"]
            .str.split("/")
            .apply(lambda parts: f"{source}_" + "_".join(parts[-3:]) + ext)
        )

    return records


def collect_lmu_sportscar_files(raw_root: Union[str, Path], source: str) -> pd.DataFrame:
    """
    Collects LMU Sportscar dataset metadata and labels, merges them, and processes
    some combined columns for ECG diagnoses.

    Args:
        raw_root (Union[str, Path]): Root directory containing dataset files.
        source (str): Dataset source identifier (currently unused in function).

    Returns:
        pd.DataFrame: Merged and processed DataFrame of records and labels.
                      Returns None if required files are not found.
    """

    raw_root = Path(raw_root)  # Convert if it's a string

    metadata = next(raw_root.glob("*pseud.xlsx"), None)
    labels = next(raw_root.glob("*Labeled.csv"), None)

    if metadata is None:
        print(f"Error: No metadata file found in {raw_root}.")
        return

    if labels is None:
        print(f"Error: No label file found in {raw_root}.")
        return

    records = pd.read_excel(str(metadata), index_col=0)
    labels = pd.read_csv(str(labels), index_col=0)

    records = records[records.EKG_dataMUSE != 0]

    paths = [str(raw_root / f"{i}.csv") for i in records.index]
    records["path"] = paths
    records["save_file"] = [p.split("/")[-1] for p in paths]  # here it is the same as it is .csv files
    records = records.join(labels)
    with pd.option_context("future.no_silent_downcasting", True):
        records = records.replace({"Ja": 1, "Nein": 0}).infer_objects(copy=False)

    column_merges = {
        "CLBBB|LBBB": ["CLBBB", "LBBB"],
        "CRBBB|RBBB": ["CRBBB", "RBBB"],
        "PAC|SVPB": ["PAC", "SVPB"],
        "PVC|VPB": ["PVC", "VPB"],
    }

    for new_col, old_col in column_merges.items():
        records[new_col] = records[old_col].any(axis=1).astype(int)
        records.drop(old_col, errors="ignore", inplace=True, axis=1)

    return records


def collect_norwegian_files(raw_root: Union[str, Path], source: str) -> pd.DataFrame:
    """
    Wrapper function to collect Norwegian athlete ECG dataset files. Dataset is from Physionet...

    Args:
        raw_root (Union[str, Path]): Root directory containing the Norwegian dataset files.
        source (str): Dataset source identifier (e.g., 'norwegian-athlete').

    Returns:
        pd.DataFrame: DataFrame containing file paths and metadata as returned by collect_physionet_files.
    """
    return collect_physionet_files(raw_root, source)


def collect_code15_files(data_dir: Union[str, Path], source: str) -> pd.DataFrame:
    """
    Collects records for the CODE-15 dataset.
    for now, we only use part0 for the evaluation.
    """
    data_dir = Path(data_dir)
    
    csv_path = data_dir / "exams.csv" # meta data table
    if not csv_path.exists():
        raise FileNotFoundError(f"CODE-15 metadata not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    df = df[df["trace_file"] == "exams_part0.hdf5"].copy() # filter patients in part0, can adjust this later on
    
    hdf5_path = data_dir / "exams_part0.hdf5"
    
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": str(row["exam_id"]),
            "path": "exams_part0.hdf5::" + str(row["exam_id"]),
            "save_file": f"{row['exam_id']}.mat",
            "metadata": row.to_dict()
        })
        
    return pd.DataFrame(records)