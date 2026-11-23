from typing import Tuple, Any, Dict
import wfdb
import numpy as np
from scipy.io import savemat, loadmat
from scipy.io.matlab.mio5_params import mat_struct
import pandas as pd
import os
import h5py


def load_physionet(path: str) -> Tuple[np.ndarray, dict]:
    """
    Load a PhysioNet signal record using WFDB, returning signal as (samples x channels).

    Args:
        path : str
            Base path to the record (without file extension).
            Requires both .hea and .mat files.

    Returns:
        Tuple[np.ndarray, dict]
            - Signal data as a NumPy array with shape (samples x channels).
            - Metadata dictionary from the .hea file
    """
    signals, fields = wfdb.rdsamp(path)
    return signals, fields


def load_lmu_sportscar(path: str) -> Tuple[np.ndarray, dict]:
    """
    Load a LMU Sportscar signal record using Pandas, returning signal as (samples x channels).

    Args:
        path : str
            Base path to the record (without file extension).
            Requires .csv files.

    Returns:
        Tuple[np.ndarray, dict]
            - Signal data as a NumPy array with shape (samples x channels).
    """
    data = pd.read_csv(path)
    signals = data.iloc[:, :12].to_numpy().astype(np.float64)
    fields = {"fs": 500}
    fields["sig_name"] = [col.strip() for col in data.columns[:12]]
    fields["sig_len"] = signals.shape[1]
    return signals, fields


def load_norwegian(path: str) -> Tuple[np.ndarray, dict]:
    """
    Load Norwegian athlete ECG data from the specified path. Wrapper for physionet.
    Args:
        path (str): File path to the Norwegian dataset file.

    Returns:
        Tuple[np.ndarray, dict]: Tuple containing the ECG signal array and associated metadata dictionary.
    """
    return load_physionet(path)


def save_file(path: str, data: Dict[str, Any]) -> None:
    """
    Save data to a .mat file using SciPy's savemat, with error handling.

    Args:
        path : str
            The file path where the .mat file will be saved.

        data : dict
            A dictionary where keys are variable names and values are the data to save.

    Raises:
        ValueError
            If the data is not in the expected dictionary format.

        OSError
            If writing the file fails due to I/O or permission issues.
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary of variable names to values.")

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        savemat(path, data)

    except OSError as e:
        raise OSError(f"Failed to write .mat file to '{path}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while saving to '{path}': {e}") from e


def load_file(path: str) -> Dict[str, Any]:
    """
    Load data from a .mat file and convert all mat_structs to native Python dicts.

    Args:
        path : str
            The path to the .mat file.

    Returns:
        Dict[str, Any]
            A dictionary mirroring the original structure saved with savemat.

    Raises:
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file content is not in the expected format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MAT file not found at '{path}'.")

    try:
        raw = loadmat(path, struct_as_record=False, squeeze_me=True)
        cleaned = _convert_mat_structs(raw)
        return cleaned
    except Exception as e:
        raise ValueError(f"Failed to load and parse MAT file '{path}': {e}") from e


def _convert_mat_structs(obj: Any) -> Any:
    """
    Recursively convert MATLAB mat_struct objects loaded by scipy.io.loadmat
    into nested Python dictionaries and convert arrays of mat_structs into lists of dicts.

    Args:
        obj (Any): Object loaded from a .mat file, potentially containing mat_structs.

    Returns:
        Any: The input object converted into native Python types (dicts, lists, or numpy arrays).
    """
    if isinstance(obj, mat_struct):
        return {field: _convert_mat_structs(getattr(obj, field)) for field in obj._fieldnames}
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_convert_mat_structs(el) for el in obj]
        else:
            return obj  # keep numeric arrays as-is
    elif isinstance(obj, dict):
        return {
            k: _convert_mat_structs(v)
            for k, v in obj.items()
            if not k.startswith("__")  # skip MATLAB metadata
        }
    else:
        return obj

def load_code15(path_and_id: str):
    path_and_id = str(path_and_id)
    file_path, exam_id = path_and_id.split("::")
    exam_id = int(exam_id)
    
    with h5py.File(file_path, 'r') as f:
        # unclear names in documentation
        id_key = 'exam_id' if 'exam_id' in f else 'id_exam'
        sig_key = 'tracings' if 'tracings' in f else 'signal'
        
        ids = f[id_key][:]
        idx = np.where(ids == exam_id)[0][0]
        
        # (4096, 12) rotate to (12, 4096)
        sig_array = f[sig_key][idx].T
        
    fields = {
        "fs": 400,
        "sig_len": 4096,
        "sig_name": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    }
    
    return sig_array, fields