from torch.utils.data import DataLoader
import pandas as pd
from typing import Literal

from ecgath.data.dataset import ECGFMDataset, ECGFounderDataset, HubertECGDataset, LeadSelect

def create_dataloader(
    file_list: pd.DataFrame,
    base_path: str, 
    batch_size: int = 32, 
    split: str = None, 
    model_type: Literal['ecgfm', 'ecgfounder', 'hubertecg'] = 'ecgfm', 
    lead: int = 0,
):

    """
    Create a DataLoader from a file list of .mat files.

    Args:
        file_list (pandas.DataFrame): DataFrame with file paths.
        base_path (str): The base path of all files.
        batch_size (int): Batch size.
        split (str): Optional split to filter the file list by ('train', 'val', 'test').
        model_type (Literal): Specifies which dataset class to use.

    Returns:
        DataLoader: A DataLoader instance.
    """
    
    if split:
        file_list = file_list[file_list.split == split]
    
    if file_list.empty: # add
        return None     # for evaluation where we only have a testfile
    
    if model_type == 'ecgfm':
        dataset = ECGFMDataset(file_list, base_path)
    elif 'ecgfounder' in model_type:
        dataset = ECGFounderDataset(file_list, base_path)
        if "1lead" in model_type:
            dataset = LeadSelect(dataset, leads=(lead,)) 
    elif 'hubert' in model_type:
        dataset = HubertECGDataset(file_list, base_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    should_shuffle = (split == 'train')    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle)

    return dataloader