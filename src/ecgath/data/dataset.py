import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from ecgath.data.io import load_file


class BaseECGDataset(Dataset):
    def __init__(self, file_list: pd.DataFrame, base_path: str):
        if "save_path" not in file_list.columns:
            raise ValueError("DataFrame must contain a 'save_path' column.")
        
        file_list = file_list.copy()
        file_list['save_path'] = file_list['save_path'].apply(lambda x: os.path.join(base_path, x))
        self.file_list = file_list["save_path"].tolist()
        
        if not self.file_list:
            raise ValueError("No .mat files found in the provided file list.")

    def __len__(self):
        return len(self.file_list)

    def _load_data(self, idx):
        path = self.file_list[idx]
        
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        mat = load_file(path)
        if "sig_array" not in mat:
            raise KeyError(f"'sig_array' key not found in {path}")
        if "labels" not in mat:
            raise KeyError(f"'labels' key not found in {path}")

        signals = torch.tensor(mat["sig_array"], dtype=torch.float32)

        label_dict = mat["labels"]
        sorted_label_dict = dict(sorted(label_dict.items()))
        label_values = list(sorted_label_dict.values())

        # assumes labels are numeric
        labels = torch.tensor(label_values, dtype=torch.float32)

        # ESSENTIAL ADDITION: Extract the segment ID (filename without .mat)
        segment_id = os.path.basename(path).replace('.mat', '')

        # Return the segment_id - essential to calculate athlete aggregated metrics
        return {"signals": signals, "labels": labels, "ecg_id": segment_id}

    def __getitem__(self, idx):
        return self._load_data(idx)


class ECGFMDataset(BaseECGDataset):
    pass


class ECGFounderDataset(BaseECGDataset):
    pass


class HubertECGDataset(BaseECGDataset):
    def _compute_attention_mask_for_padding(self, array: torch.Tensor):
        """
        array: shape (channels, time)
        Returns attention mask of shape (channels * time,)
        """
        non_zero_cols = torch.any(array != 0, dim=0)
        start = torch.where(non_zero_cols)[0][0]
        end = torch.where(non_zero_cols)[0][-1]

        attention_mask = torch.zeros(array.shape[1], dtype=torch.long, device=array.device)
        attention_mask[start:end + 1] = 1

        # repeat mask per channel
        attention_mask = attention_mask.repeat(array.shape[0])
        return attention_mask

    def __getitem__(self, idx):
        data = self._load_data(idx)
        signals = data["signals"]  # (channels, time)

        attention_mask = self._compute_attention_mask_for_padding(signals)

        data.update({
            "signals": signals.reshape(-1),
            "attention_mask": attention_mask
        })
        return data

class LeadSelect(Dataset):
    """Wraps any ECG dataset and keeps only the given lead(s)."""
    def __init__(self, base: Dataset, leads=(0,)):   # 0 = lead I
        self.base = base
        self.leads = list(leads)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = dict(self.base[idx])          
        sample["signals"] = sample["signals"][self.leads, :]   # (12, L) -> (1, L)
        return sample