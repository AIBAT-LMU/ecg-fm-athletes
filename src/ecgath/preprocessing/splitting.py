import numpy as np
import pandas as pd
from skmultilearn.model_selection import IterativeStratification
from typing import Union, List, Tuple, Optional


def split_stratified(
    meta: pd.DataFrame,
    fractions: Union[List[float], Tuple[float, ...]] = (0.8, 0.1, 0.1),
    labels: Optional[List[str]] = ['train', 'val', 'test'],
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Performs a stratified split on a multilabel dataset.
    Ensures that the distribution of labels is similar across all splits.

    Args:
        meta (pd.DataFrame): The input DataFrame to split. Must contain label columns.
        fractions (list or tuple): Fractions for train, val, and test splits.
        labels (list of str): Names for the splits.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A long-format DataFrame with split information.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Fractions must sum to 1.0")
    if len(fractions) != 3:
        raise ValueError("This function is designed for 3 splits (train, val, test).")

    label_cols = meta.filter(like='label').columns.tolist()
    if not label_cols:
        raise ValueError("No 'label_' columns found in the dataframe for stratification.")

    # handle samples with no labels
    meta_copy = meta.copy()  # Work on copy to not modify the original data
    label_sums = meta_copy[label_cols].sum(axis=1)
    if (label_sums == 0).any():
        print("Warning: Found samples with no labels. Temporarily assigning 'label_NORM' to proceed.")
        if 'label_NORM' not in meta_copy.columns:
            meta_copy['label_NORM'] = 0  # add column if it doesn't exist
        meta_copy.loc[label_sums == 0, 'label_NORM'] = 1

    X = meta_copy.index.to_numpy().reshape(-1, 1)
    y = meta_copy[label_cols].to_numpy()

    # 1. split into train and a temporary 'test_val' set.
    train_frac = fractions[0]
    test_val_frac = 1.0 - train_frac

    stratifier_1 = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[test_val_frac, train_frac],
        shuffle=True,
        random_state=random_state
    )
    train_indices, test_val_indices = next(stratifier_1.split(X, y))

    meta_copy.loc[meta_copy.index[train_indices], 'split'] = labels[0]

    # split temporary set into final validation and test sets.
    if len(test_val_indices) > 0:
        X_test_val = X[test_val_indices]
        y_test_val = y[test_val_indices]

        val_frac_relative = fractions[1] / (fractions[1] + fractions[2])
        test_frac_relative = 1.0 - val_frac_relative

        stratifier_2 = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[test_frac_relative, val_frac_relative],
            shuffle=True,
            random_state=random_state
        )
        
        val_indices_rel, test_indices_rel = next(stratifier_2.split(X_test_val, y_test_val))
        
        test_indices = test_val_indices[test_indices_rel]
        val_indices = test_val_indices[val_indices_rel]

        meta_copy.loc[meta_copy.index[val_indices], 'split'] = labels[1] 
        meta_copy.loc[meta_copy.index[test_indices], 'split'] = labels[2]

    save_path_cols = meta_copy.columns[meta_copy.columns.str.startswith('save_path')]

    if len(save_path_cols) == 1 and save_path_cols[0] == 'save_path':
        split_df = meta_copy[['split', 'save_path'] + label_cols]
    else:
        split_df = pd.melt(
            meta_copy,
            id_vars=['split'] + label_cols,
            value_vars=save_path_cols,
            value_name='save_path'
        )

    return split_df.dropna(subset=['save_path'])


# function for randomized splitting

def split_random(
    meta: pd.DataFrame,
    fractions: Union[List[float], Tuple[float, ...]] = (0.8, 0.1, 0.1),
    labels: Optional[List[str]] = ['train', 'val', 'test'],
    random_state: int = 42,
) -> pd.Series:
    """
    Randomly splits a DataFrame into labeled splits. Accounts for file level,
    i.e., assigns the same label to the same ECG-file 
    to 

    Args:
        meta (pd.DataFrame): The input DataFrame to split.
        fractions (list or tuple of float, optional): Fractions for each split. Must sum to 1.0.
        labels (list of str, optional): Labels corresponding to each split fraction.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A long-format DataFrame with columns ['split', 'save_path'].
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Fractions must sum to 1.0")
    if labels and len(labels) != len(fractions):
        raise ValueError("Labels must match number of fractions")

    n = len(meta)
    idx = np.random.RandomState(random_state).permutation(n)
    split_points = np.round(np.cumsum(fractions[:-1]) * n).astype(int)
    split_indices = np.split(idx, split_points)

    if labels is None:
        labels = [f"split_{i}" for i in range(len(fractions))]

    for indices, label in zip(split_indices, labels):
        meta.loc[indices, 'split'] = label

    save_path_cols = meta.columns[meta.columns.str.startswith('save_path')]
    label_cols = meta.filter(like='label').columns.tolist()
    split_df = meta[list(save_path_cols) + ['split'] + label_cols]

    split_df = pd.melt(
        split_df,
        id_vars=['split'] + label_cols,
        value_vars=save_path_cols,
        value_name='save_path'
    )

    split_df = split_df[['split', 'save_path'] + label_cols]

    return split_df

# several safety steps included to ensure robustness
def get_label_weights(splitted: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and returns a DataFrame with class weights and distributions
    to verify the quality of the data split.

    """
    working_df = splitted.copy()
    
    # ensure columns that should be numeric are numeric
    label_cols = [col for col in working_df.columns if col.startswith('label_')]
    for col in label_cols:
        working_df[col] = pd.to_numeric(working_df[col], errors='coerce').fillna(0)

    working_df.columns = working_df.columns.str.replace('label_', '')
    label_cols = [col.replace('label_', '') for col in label_cols]

    if 'split' not in working_df.columns:
        raise ValueError("'split' column not found. The data must be split first.")
        
    # calculate counts and percentages per split
    label_counts = working_df.groupby('split')[label_cols].sum()
    split_sizes = working_df.groupby('split')['save_path'].nunique()

    # 'errors='coerce'' will turn any non-numeric value into NaN,
    # and .fillna(0) will replace those NaNs with 0.
    split_sizes = pd.to_numeric(split_sizes, errors='coerce').fillna(0)
   
    # .astype(float) to prevent potential division by zero errors if a split is empty
    label_percents = label_counts.div(split_sizes.astype(float), axis=0).fillna(0)
    
    # avoid division by zero for weights
    with np.errstate(divide='ignore', invalid='ignore'):
        label_weights = (1 - label_percents) / label_percents
    label_weights.replace([np.inf, -np.inf], 0, inplace=True) # replace inf with 0
    label_weights.fillna(0, inplace=True)

    # format the output DataFrame
    label_counts_prefixed = label_counts.T.add_prefix('pos_count_')
    label_percents_prefixed = label_percents.T.add_prefix('pos_percent_')
    label_weights_prefixed = label_weights.T.add_prefix('pos_weight_')

    label_def = pd.concat([
        label_counts_prefixed,
        label_percents_prefixed,
        label_weights_prefixed,
    ], axis=1).sort_index()

    # calculate statistics for entire dataset
    labels_df_all = working_df.drop(columns=['split', 'save_path'])
    label_def['pos_count_all'] = labels_df_all.sum(axis=0)
    
    total_unique_samples = splitted['save_path'].nunique()
    if total_unique_samples > 0:
        label_def['pos_percent_all'] = label_def['pos_count_all'] / total_unique_samples
    else:
        label_def['pos_percent_all'] = 0

    return label_def.fillna(0)