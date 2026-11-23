
from typing import Union, List, Optional
from pathlib import Path
import pandas as pd
import os

def remove_ext(path: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Removes the file extension from a file path or a list of file paths.

    Parameters
    ----------
    path : Union[str, List[str]]
        A single file path as a string or a list of file paths.

    Returns
    -------
    Union[str, List[str]]
        The path(s) with the file extension removed.
    """
    if isinstance(path, str):
        return str(Path(path).with_suffix(''))

    if isinstance(path, list):
        return [str(Path(p).with_suffix('')) for p in path]

    raise TypeError("Input must be a string or a list of strings.")


def remove_common_segments(paths: List[str]) -> List[str]:
    """
    Removes the shared directory prefix from a list of file paths.

    Parameters
    ----------
    paths : List[str]
        A list of file paths.

    Returns
    -------
    List[str]
        Paths with the shared leading directory segments removed.
    """
    if not paths:
        return []

    try:
        common_path = os.path.commonpath(paths)
    except ValueError:
        return paths

    return [os.path.relpath(p, common_path) for p in paths]


def filenames_from_paths(
    paths: pd.Series,
    replacement_ext: Optional[str] = None,
    exclude_common_segments: bool = True,
    exclude_common_prefix: bool = False,
    warn_not_unique: bool = True,
):
    """
    Generate filenames from file paths.

    Useful for creating unique filenames from file paths, but this is not guaranteed,
    therefore a warning is issued by default.

    Parameters
    ----------
    paths : pd.Series
        A series containing file paths.
    replacement_ext : str, optional
        The new extension to replace existing extensions, no replacement by default.
    exclude_common_segments : bool, default True
        If True, removes the first directory segments shared by all of the files.
    exclude_common_prefix : bool, default False
        If True, excludes the common prefix from each filename to reduce filename lengths.
    warn_not_unique : bool, default True
        Whether to raise a warning if the resulting filenames are not unique.

    Returns
    -------
    pd.Series
        A series of generated filenames derived from the input file paths.

    Warnings
    --------
    Warn when warn_not_unique is True and the resulting filenames are not unique.
    """
    if exclude_common_segments and exclude_common_prefix:
        raise ValueError(
            "Cannot specify `exclude_common_segments` and `exclude_common_prefix`."
        )

    if replacement_ext is not None:
        # Replace extensions
        paths = replace_ext(paths, replacement_ext)

    if exclude_common_segments:
        paths = remove_common_segments(paths)

    if exclude_common_prefix:
        paths = remove_common_prefix(paths)

    # Convert from a path into a viable filename
    paths = paths.str.replace("/", "_").str.replace(" ", "_")

    if warn_not_unique and not paths.is_unique:
        warnings.warn("Resulting filenames are not unique.")

    return paths