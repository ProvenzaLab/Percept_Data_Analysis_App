import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from zoneinfo import ZoneInfo
from datetime import timedelta
import os, sys, shutil
from appdirs import user_data_dir

central_time = ZoneInfo("America/Chicago")


def daily_operation(group, cols_to_operate, new_col_names, operation=pd.Series.var):
    new_df = pd.DataFrame(index=group.index, columns=new_col_names)
    for col, new_col in zip(cols_to_operate, new_col_names):
        new_df[new_col] = operation(group[col])
    return new_df

def get_data_path(subdir='data'):
    """Return the path to the app's data directory, ensuring it's writable."""
    if hasattr(sys, '_MEIPASS'):
        # Running from a PyInstaller bundle
        base_path = os.path.join(sys._MEIPASS, subdir)
    else:
        # Running from source
        base_path = os.path.abspath(subdir)

    # Create a user-writable copy in the local directory
    user_data_dir = os.path.join(os.getcwd(), subdir)
    if not os.path.exists(user_data_dir):
        shutil.copytree(base_path, user_data_dir)
    return user_data_dir

def zscore_group(group, cols_to_zscore=["lfp_left_raw", "lfp_right_raw"]):
    """
    Calculate Z-scored version of specified data.

    Parameters:
    - group (pd.DataFrame): group dataframe to Z-score across.
    - cols_to_zscore (list): columns to calculate and return Z-scored version of.

    Returns:
    - cols_z_scored_df (pd.DataFrame): DataFrame containing only Z-scored version of provided columns. Output is NaN wherever input was NaN.
    """
    new_cols = {}
    for col in cols_to_zscore:
        # Write new column name for easy merging later.
        if "_raw" in col:
            zscored_col_name = col.replace("_raw", "_z_scored")
        elif "_filled" in col:
            zscored_col_name = col.replace("_filled", "_z_scored")
        else:
            zscored_col_name = col + "_z_scored"

        if pd.notna(group[col]).sum() > 1:
            # Z-score data: for each data point, subtract mean and divide by standard deviation of entire series.
            mean = np.nanmean(group[col], axis=0)
            std = np.nanstd(group[col], axis=0, ddof=1)
            new_cols[zscored_col_name] = (
                (group[col] - mean) / std if std != 0 else [np.nan] * len(group[col])
            )
        else:
            # If all values are none, return np.nan
            new_cols[zscored_col_name] = [np.nan] * len(group)

    # Create new dataframe containing only the new Z-scored columns and return it. Keep the same indices as the input group for easy merging.
    return pd.DataFrame(new_cols, index=group.index)


def correct_timestamps(
    group, timestamp_col="timestamp", threshold=pd.Timedelta(minutes=10, seconds=1)
):
    """
    Adjust timestamps in each group where intervals are off by one second.

    Parameters:
    - group: DataFrame containing a column with timestamps.
    - timestamp_col: Name of the dataframe column containing the timestamp.
    - threshold: Maximum expected time difference for a 10-minute interval (default: 10:01).

    Returns:
    - Timestamp column with corrected timestamps.
    """
    timestamps = group[timestamp_col].sort_values()
    time_diff = timestamps.diff()

    correction_mask = time_diff == threshold
    reset_mask = time_diff > threshold
    correction_groups = reset_mask.cumsum()
    correction = correction_mask.groupby(correction_groups).cumsum() * -pd.Timedelta(
        seconds=1
    )

    return pd.DataFrame(timestamps + correction)


def get_contig(group, col_to_check, contig_colname, time_bin_col="time_bin"):
    """
    Label each row in a contiguous series with the same number, and return a DataFrame containing those labels.
    Contiguous data is a series of data where each data point is separated by ~10 minutes (with margin of error 1 second either direction).

    Parameters:
    - group (pd.DataFrame): DataFrame to get contiguous data labels of.
    - col_to_check (str): Name of the column you are interested in getting contiguous data labels for. Rows containing NaN values in this column will be dropped prior to generating labels.
    - contig_colname (str): Name of the column in the returned DataFrame containing contiguous data labels.
    - time_bin_col (str): Name of the column containing data time bin timestamps.

    Returns:
    - contig_df (pd.DataFrame): DataFrame containing only labels for sections of contiguous data, where indices match indices of the input.
    """
    # Drop any NaN values of the column to get contiguous labels for.
    group = group.dropna(subset=col_to_check)

    # Ensure values are sorted by timestamp.
    group = group.sort_values(time_bin_col)

    time_diff = group[time_bin_col].diff()
    contig = (time_diff != timedelta(minutes=10)).cumsum() - 1

    return pd.DataFrame({contig_colname: contig}, index=group.index, dtype="Int64")


def fill_holes(group, lead_location, time_bin_col="time_bin", dbs_on_date=None):
    """
    Fills in dataframe holes so that all missing time bins contain NaN. Useful for interpolation later.

    Parameters:
    - group (pd.DataFrame): DataFrame containing a single patient's data from a single lead.
    - lead_location (str): Lead location for this lead.
    - time_bin_col (str, optional): Name of the DataFrame column containing the datas' time bins timestamps.
    - dbs_on_date (datetime, optional): Date when the DBS was turned on. If provided, will fill the "days_since_dbs" column.

    Returns:
    - interp_df (pd.DataFrame): DataFrame containing empty rows where no data was recorded by the Percept device.
    """

    # Get sizes of gaps of missing data in terms of number of missing data points.
    gap_sizes = (np.diff(group[time_bin_col]) // timedelta(minutes=10)).astype(int)
    small_gap_start_inds = np.where(gap_sizes >= 2)[0]
    gap_sizes = gap_sizes[small_gap_start_inds]

    # Get the last time bin timestamp before each unfilled data gap so we know where to start filling from.
    gap_start_times = group.loc[group.index[small_gap_start_inds], time_bin_col]
    times_to_fill = [
        gap_start_time + timedelta(minutes=10) * i
        for (gap_start_time, gap_size) in zip(gap_start_times, gap_sizes)
        for i in range(1, gap_size)
    ]

    # Create new dataframe and fill in information in relevant columns.
    if len(times_to_fill) != 0:
        interp_df = pd.DataFrame()
        interp_df["timestamp"] = times_to_fill  # Timestamp is set to time bin
        interp_df[time_bin_col] = (
            times_to_fill  # Time bin is where the data was not recorded/missing from the device.
        )
        interp_df["CT_timestamp"] = interp_df["timestamp"].dt.tz_convert(central_time)
        if dbs_on_date is not None:
            interp_df["days_since_dbs"] = [
                dt.days for dt in (interp_df["CT_timestamp"].dt.date - dbs_on_date)
            ]
        interp_df["lead_location"] = (
            lead_location  # Use same lead model and location as original df.
        )
        interp_df["lead_model"] = np.repeat(
            group.loc[group.index[small_gap_start_inds], "lead_model"].values,
            gap_sizes - 1,
        )
        interp_df["source_file"] = (
            "interp"  # Denote filled rows as interpolated so we know they aren't real data.
        )
        interp_df["interpolated"] = True
        return interp_df


def fill_outliers(data: np.ndarray, threshold_factor: int = 30) -> np.ndarray:
    """
    Replace outliers in the data with interpolated values using PCHIP interpolation.

    Parameters:
        data (np.ndarray): Input data array with potential outliers.
        threshold_factor (int, optional): Factor to define the threshold for outliers. Default is 30.

    Returns:
        np.ndarray: Data array with outliers filled.
    """
    not_nan = ~np.isnan(data)
    if not_nan.sum() < 2:
        return np.empty_like(data) * np.nan
    median = np.median(data[not_nan])
    MATLAB_MAD_SCALE_FACTOR = 1.4826
    mad = np.median(np.abs(data[not_nan] - median)) * MATLAB_MAD_SCALE_FACTOR
    threshold = threshold_factor * mad
    outliers = (np.abs(data - median) > threshold) & not_nan
    valid_indices = np.where(not_nan & ~outliers)[0]
    valid_values = data[valid_indices]
    interpolator = PchipInterpolator(valid_indices, valid_values)
    corrected_data = data.copy()
    corrected_data[outliers] = interpolator(np.where(outliers)[0])
    return corrected_data


def fill_outliers_threshold(
    data: np.ndarray, threshold: float = ((2**32) - 1) / 60
) -> np.ndarray:
    """
    Replace outliers in the data with interpolated values using PCHIP interpolation.

    Parameters:
        data (np.ndarray): Input data array with potential outliers.
        threshold (float, optional): Threshold to define the outliers. Default is max int value divided by 60.

    Returns:
        np.ndarray: Data array with outliers filled.
    """
    # Strip data of any nans.
    not_nan = ~np.isnan(data)
    if not_nan.sum() < 2:
        return np.empty_like(data) * np.nan

    # Find indices of outliers.
    outliers = (data >= threshold) & not_nan

    # Set up the PCHIP interpolator.
    valid_indices = np.where(not_nan & ~outliers)[0]
    valid_values = data[valid_indices]
    interpolator = PchipInterpolator(valid_indices, valid_values)
    corrected_data = data.copy().astype(float)
    corrected_data[outliers] = np.nan

    # Don't interpolate any trailing or leading outlier values.
    if outliers[-1]:
        outliers[-np.argmax(outliers[::-1] == False) :] = False
    if outliers[0]:
        outliers[: np.argmax(outliers == False)] = False

    # Interpolate the outliers using PCHIP.
    corrected_data[outliers] = interpolator(np.where(outliers)[0])
    return corrected_data


def fill_outliers_overages(data: np.ndarray):
    """
    Replace outliers in the data caused by overvoltage readings. When the Percept device records a LFP value above its
    acceptable range, it places the maximum integer value in its place. Then, when the 10 minute interval is averaged,
    the abnormally high value dominates the average and causes non-physical outliers in the data. When multiple overages
    are observed in a single 10 minute interval, the outlier is even higher. Here, we estimate how many overages were
    recorded during each 10 minute interval, then remove them and recalculate the averaged LFP without the abnormal values.
    The overvoltage recordings may be caused by movement artifacts or something else.

    Parameters:
        data (np.ndarray): 1-dimensional array containing contiguous LFP data, potentially with outliers and holes.

    Returns:
        new_data (np.ndarray): Newly calculated LFP values with outliers removed.
    """
    n = 60  # Number of samples per 10 minute average
    x = 2**32 - 1

    num_overages = data // (
        x / n
    )  # Estimate how many voltage overages we had during each 10 minute interval

    # If all samples within the interval are overages, place a NAN in. This will be filled in later when the missing values are filled.
    valid_mask = num_overages < n
    corrected_data = np.empty_like(data, dtype=float)
    corrected_data[valid_mask] = (
        n * data[valid_mask] - x * num_overages[valid_mask]
    ) / (n - num_overages[valid_mask])
    corrected_data[~valid_mask] = np.nan

    return corrected_data


def fill_missing(data: np.ndarray, max_gap: int = 7) -> np.ndarray:
    """
    Fill missing values (NaNs) in the data array using PCHIP interpolation, for gaps up to max_gap size.

    Parameters:
        data (np.ndarray): Input data array with missing values (NaNs).
        max_gap (int, optional): Maximum gap size to fill. Default is 7.

    Returns:
        np.ndarray: Data array with missing values filled.
    """
    data = data.copy()
    isnan = np.isnan(data)
    if (~isnan).sum() < 2:
        return np.empty_like(data) * np.nan
    nan_indices = np.where(isnan)[0]

    if len(nan_indices) == 0:
        return data  # No NaNs to fill

    gaps = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)
    if isnan[-1]:
        gaps = gaps[:-1]
    not_nan_indices = np.where(~isnan)[0]
    not_nan_values = data[not_nan_indices]
    interpolator = PchipInterpolator(not_nan_indices, not_nan_values)

    for gap in gaps:
        if len(gap) > 0:
            gap_size = (
                len(gap) + 2
                if (gap[0] > 0 and gap[-1] < len(data) - 1)
                else len(gap) + 1
            )
            if gap_size <= max_gap + 1:
                data[gap] = interpolator(gap)

    return data


def fill_data(group_df, cols_to_fill, outlier_fill_method, offset_col="days_since_dbs"):
    """
    Fill missing data in the DataFrame using specified methods for outlier filling.

    Parameters:
        group_df (pd.DataFrame): DataFrame containing the data to be filled.
        cols_to_fill (list): List of column names to fill.
        outlier_fill_method (callable): Function to fill outliers.
        offset_col (str): Column name for the days since DBS.
    """
    all_new_cols = []
    for col in cols_to_fill:
        group_df_no_na = group_df.dropna(subset=[col])

        start_index = np.where(group_df_no_na[offset_col].diff() > 1)[0]
        start_index = np.array(
            [0]
            + [group_df_no_na.index[i] for i in start_index]
            + [group_df.index.max() + 1]
        )

        new_col_name = col.replace("_raw", "")
        new_cols = pd.DataFrame(
            np.nan,
            columns=[f"{new_col_name}_outliers_filled", f"{new_col_name}_filled"],
            index=group_df.index,
        )

        for i in range(len(start_index) - 1):
            if i == 3:
                pass
            data = group_df.loc[start_index[i] : start_index[i + 1] - 1, col]

            if np.all(np.isnan(data)):
                continue
            data_no_trailing_nans = data[
                : np.where(~np.isnan(data))[0][-1] + 1
            ]  # Remove trailing NaNs

            outliers_filled = outlier_fill_method(data_no_trailing_nans.values)
            missing_filled = fill_missing(outliers_filled)
            new_cols.loc[
                data_no_trailing_nans.index, f"{new_col_name}_outliers_filled"
            ] = outliers_filled
            new_cols.loc[data_no_trailing_nans.index, f"{new_col_name}_filled"] = (
                missing_filled
            )

        all_new_cols.append(new_cols)

    return pd.concat(all_new_cols, axis=1)


def get_sig_text(pval):
    if pval < 0.0001:
        return "****"
    elif pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    return "ns"


def resource_path(relative_path):
    # Works for development and PyInstaller
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
