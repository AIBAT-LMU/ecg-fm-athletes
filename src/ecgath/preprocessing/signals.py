import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Literal, Any
from scipy.signal import butter, filtfilt, iirnotch, firwin
from scipy.interpolate import interp1d

LEAD_ORDER: Dict[int, str] = {
    0: "I",
    1: "II",
    2: "III",
    3: "aVR",
    4: "aVL",
    5: "aVF",
    6: "V1",
    7: "V2",
    8: "V3",
    9: "V4",
    10: "V5",
    11: "V6",
}


def reshape(sig_array: Union[np.ndarray, list]) -> np.ndarray:
    """
    Ensure the signal is shaped as (channels, samples). If the first dimension
    is not the number of channels, attempt to transpose it.

    Parameters
    ----------
    signal : Union[np.ndarray, list]
        A 2D array or list-like containing signal data.

    Returns
    -------
    np.ndarray
        Signal reshaped to (channels, samples).
    """
    sig_array = np.asarray(sig_array)

    if sig_array.ndim != 2:
        raise ValueError(f"Expected a 2D signal, got shape {sig_array.shape}")

    channels, samples = sig_array.shape

    if channels > samples:
        return sig_array.T
    return sig_array


def reorder_leads(
    sig_array: np.ndarray, sig_names: List[str], lead_indices: Optional[List[int]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Reorder `sig_array` to match a subset of LEAD_ORDER.

    Parameters
    ----------
    sig_array : np.ndarray
        Array of shape (channels, samples).
    sig_names : list of str
        List of lead names matching the rows of `sig_array`.
    lead_indices : list of int, optional
        Indices from LEAD_ORDER to select (default: all leads).

    Returns
    -------
    reordered_sig_array : np.ndarray
        Array of shape (len(lead_indices), samples), ordered and padded with NaNs for missing leads.
    available_leads : list of str
        List of leads present in input and reordered accordingly.
    """
    sig_names = list(sig_names)
    name_to_input_index = {name: i for i, name in enumerate(sig_names)}

    if lead_indices is None:
        lead_indices = list(LEAD_ORDER.keys())

    selected_leads = {i: LEAD_ORDER[i] for i in lead_indices}

    reordered_sig_array = np.full((len(selected_leads), sig_array.shape[1]), np.nan)
    available_leads = []

    for idx, lead_name in selected_leads.items():
        if lead_name in name_to_input_index:
            reordered_sig_array[idx] = sig_array[name_to_input_index[lead_name]]
            available_leads.append(lead_name)

    return reordered_sig_array, available_leads


def extract_sig_info(sig_array: np.ndarray) -> Dict:
    """
    Extract summary statistics and quality flags for each lead in the signal array.

    Parameters
    ----------
    sig_array : np.ndarray
        2D array of shape (channels, samples), where each row corresponds to a lead's signal data.

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean' : np.ndarray
            Mean value of each lead's signal (shape: channels).
        - 'std' : np.ndarray
            Standard deviation of each lead's signal (shape: channels).
        - 'nan_any' : bool
            True if any NaN values exist anywhere in the array.
        - 'nan_leads_any' : bool
            True if any lead consists entirely of NaNs.
        - 'nan_leads' : list of str
            Names of leads that are fully NaN (empty signals).
        - 'constant_leads_any' : bool
            True if any lead has a constant value (no variation).
        - 'constant_leads' : list of str
            Names of leads that have constant signals.

    Notes
    -----
    Lead names are determined from the global `LEAD_ORDER` constant by index.
    """

    sig_info = {}

    mean = sig_array.mean(axis=0, keepdims=True)
    sig_info["mean"] = mean.flatten()
    sig_info["std"] = np.std(sig_array, axis=0)

    nan_mask = np.isnan(sig_array)
    nan_per_lead = nan_mask.sum(axis=1)
    fully_nan_leads = nan_per_lead == sig_array.shape[1]

    sig_info["nan_any"] = nan_mask.any()
    sig_info["nan_leads_any"] = fully_nan_leads.any()
    sig_info["nan_leads"] = [LEAD_ORDER[i] for i, is_nan in enumerate(fully_nan_leads) if is_nan]

    constant_leads = np.count_nonzero(sig_array - mean, axis=1) == 0
    sig_info["constant_leads_any"] = constant_leads.any()
    sig_info["constant_leads"] = [
        LEAD_ORDER[i] for i, is_const in enumerate(constant_leads) if is_const
    ]

    return sig_info


def apply_filter(
    sig_array: np.ndarray, fs: int, strategy: Literal["founder", "hubert", None] = None
) -> np.ndarray:
    """
    Applies the specific filter strategy used by the different models.

    Args:
        sig_array (np.ndarray): ECG signal (channels, samples).
        fs (int): sampling rate of the signal.
        strategy (str): 'founder' for the high/low/notch filter chain.
                        'hubert' for the FIR bandpass filter.
                        None to skip filtering.

    Returns:
        np.ndarray: filtered ECG signal.
    """
    if not strategy:
        return sig_array

    nyquist = 0.5 * fs

    if strategy == "founder":
        b, a = butter(2, 1 / nyquist, btype="high")
        filtered_sig_array = filtfilt(b, a, sig_array, axis=1)

        b, a = butter(2, 30 / nyquist, btype="low")
        filtered_sig_array = filtfilt(b, a, filtered_sig_array, axis=1)

        b, a = iirnotch(50, Q=30, fs=fs)
        return filtfilt(b, a, filtered_sig_array, axis=1)

    elif (
        strategy == "hubert"
    ):  # hubert origanially uses:  'from biosppy.signals.tools import filter_signal' - instead of scipy...?

        num_taps = int(0.3 * fs)

        if num_taps % 2 == 0:
            num_taps += 1

        b = firwin(num_taps, [0.05, 47], pass_zero=False, fs=fs)

        return filtfilt(b, [1.0], sig_array, axis=1)

    else:
        raise ValueError(f"Unknown filter strategy: {strategy}")


def resample(sig_array: np.ndarray, input_fs: int, output_fs: int) -> np.ndarray:
    """
    Resamples a signal to a target frequency using linear interpolation.

    Args:
        sig_array (np.ndarray): ECG signal (channels, samples).
        input_fs (int): original sampling rate.
        output_fs (int): desired sampling rate.

    Returns:
        np.ndarray: resampled signal.
    """
    if input_fs == output_fs:
        return sig_array

    num_samples_original = sig_array.shape[1]
    duration_s = num_samples_original / input_fs
    num_samples_target = int(duration_s * output_fs)

    time_original = np.linspace(0, duration_s, num_samples_original)
    time_target = np.linspace(0, duration_s, num_samples_target)

    interpolator = interp1d(
        time_original, sig_array, kind="linear", axis=1, fill_value="extrapolate"
    )  # vectorisation over all channels simult.
    return interpolator(time_target)


def standardize(
    sig_array: np.ndarray,
    strategy: Literal["z_score", "rescale", None] = None,
    constant_lead_strategy: Literal["zero", None] = None,
) -> np.ndarray:
    """
    Normalizes ECG signal.

    Args:
        sig_array (np.ndarray): ECG signal with shape (channels, samples).
        strategy (str):
            'z_score' - zero-mean, unit-variance normalization
            'rescale' - min-max scaling to [-1, 1]
            None    - no normalization
        constant_lead_strategy (str):
            'zero' - zero out constant leads (std == 0)
            None - leave them unchanged

    Returns
        np.ndarray: Normalized signal.
    """
    sig_array = sig_array.copy()
    std = np.std(sig_array, axis=1, keepdims=True)
    constant_leads = std == 0

    if strategy == "z_score":
        mean = np.mean(sig_array, axis=1, keepdims=True)
        std_safe = std.copy()
        std_safe[constant_leads] = 1
        sig_array = (sig_array - mean) / std_safe

    elif strategy == "rescale":
        min_val = np.min(sig_array, axis=1, keepdims=True)
        max_val = np.max(sig_array, axis=1, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        sig_array = 2 * ((sig_array - min_val) / range_val) - 1

    elif not strategy:
        pass

    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")

    if constant_lead_strategy == "zero":
        sig_array[constant_leads] = 0

    elif not constant_lead_strategy:
        # Leave constant leads as is
        pass

    else:
        raise ValueError(f"Unknown constant lead strategy: {constant_lead_strategy}")

    return sig_array


def segment(
    sig_array: np.ndarray,
    fs: int,
    segment_len_s: float,
    handle_remainder: Literal["pad", "drop"] = "drop",
) -> List[np.ndarray]:
    """
    Segments an ECG signal into chunks of a fixed length in seconds.

    Args:
        sig_array (np.ndarray): ECG signal (already resampled).
        fs (int): sampling rate.
        segment_len_s (float): desired segment length in seconds.
        handle_remainder (str): How to handle the last, incomplete segment.
                                'pad': Fills the last segment with zeros.
                                'drop': Throws away the last segment.

    Returns:
        List[np.ndarray]: A list of the signal segments.
    """
    segment_len_samples = int(segment_len_s * fs)
    if segment_len_samples == 0:
        return []

    num_segments = sig_array.shape[1] // segment_len_samples
    segments = [
        sig_array[:, i * segment_len_samples : (i + 1) * segment_len_samples]
        for i in range(num_segments)
    ]

    remainder_len = sig_array.shape[1] % segment_len_samples
    if remainder_len > 0 and handle_remainder == "pad":
        last_segment = sig_array[:, num_segments * segment_len_samples :]
        padding_needed = segment_len_samples - remainder_len
        padded_segment = np.pad(last_segment, ((0, 0), (0, padding_needed)), "constant")
        segments.append(padded_segment)

    return segments


def process_signal(
    sig_array: np.ndarray,
    fields: Dict[str, Any],
    desired_sample_rate: int,
    standardize_strategy: str,
    filter_strategy: str,
    constant_lead_strategy: str,
    leads: Optional[List[str]],
    segment_len_s: float,
    handle_remainder: str,
    save_path: Union[str, None],
    idx: Union[int, str],
    labels: Optional[Union[List[str], Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Processes and segments a physiological signal array according to the provided parameters.

    Args:
        sig_array (np.ndarray): Raw signal array of shape (n_channels, n_samples).
        fields (Dict[str, Any]): Metadata associated with the signal (e.g., fs, sig_name).
        desired_sample_rate (int): Target sample rate after resampling.
        standardize_strategy (str): Strategy for standardizing signal (e.g., "zscore").
        filter_strategy (str): Filtering strategy to apply (e.g., "bandpass").
        constant_lead_strategy (str): Strategy to handle missing/unavailable leads.
        leads (Optional[List[str]]): List of desired lead names (e.g., ECG leads).
        segment_len_s (float): Segment length in seconds.
        handle_remainder (str): Strategy to handle signal remainders (e.g., "discard", "pad").
        save_path (Union[str, None]): File path prefix for saving segments (may be None if unused).
        idx (Union[int, str]): Index or identifier of the sample.
        labels (Optional[Union[List[str], Dict[str, Any]]]): Associated labels or annotations.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            - List of segment data dictionaries.
            - Updated metadata fields.
    """
    current_sample_rate = fields["fs"]
    sig_array = reshape(sig_array)

    
    # first, rearrange the leads to ensure that the array has the correct number of channels (12).
    sig_array, fields["avail_leads"] = reorder_leads(sig_array, fields["sig_name"], leads)
    
    # only now extract the signal information, as the array now has the correct size.
    fields.update(extract_sig_info(sig_array))

    sig_array = apply_filter(sig_array, current_sample_rate, filter_strategy)
    sig_array = resample(sig_array, current_sample_rate, desired_sample_rate)
    current_sample_rate = desired_sample_rate
    sig_array = standardize(sig_array, standardize_strategy)

    data = {
        "org_sample_rate": fields["fs"],
        "curr_sample_rate": current_sample_rate,
        "org_sample_size": fields["sig_len"],
        "curr_sample_size": sig_array.shape[1],
        "idx": idx,
        "labels": labels,
    }

    sig_segments = segment(sig_array, desired_sample_rate, segment_len_s, handle_remainder)
    fields["n_segments"] = len(sig_segments)
    
    labels = {f'label_{key}': value for key, value in labels.items()}
    fields.update(labels)
    
    data_segments: List[Dict[str, Any]] = []
    for i, sig_seg in enumerate(sig_segments):
        segment_data = data.copy()
        segment_data["sig_array"] = sig_seg
        segment_data["curr_sample_size"] = sig_seg.shape[1]
        data_segments.append(segment_data)

    return data_segments, fields
