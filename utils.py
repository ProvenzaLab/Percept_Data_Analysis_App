import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from datetime import datetime
import json
from typing import Any, Dict, List, Tuple, Union

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
    median = np.median(data[not_nan])
    MATLAB_MAD_SCALE_FACTOR = 1.4826
    mad = np.median(np.abs(data[not_nan] - median)) * MATLAB_MAD_SCALE_FACTOR
    threshold = threshold_factor * mad
    outliers = (np.abs(data - median) > threshold) & not_nan
    valid_indices = np.where(not_nan & ~outliers)[0]
    valid_values = data[valid_indices]
    interpolator = PchipInterpolator(valid_indices, valid_values)
    data[outliers] = interpolator(np.where(outliers)[0])
    return data

def fill_missing(data: np.ndarray, max_gap: int = 7) -> np.ndarray:
    """
    Fill missing values (NaNs) in the data array using PCHIP interpolation, for gaps up to max_gap size.
    
    Parameters:
        data (np.ndarray): Input data array with missing values (NaNs).
        max_gap (int, optional): Maximum gap size to fill. Default is 7.
    
    Returns:
        np.ndarray: Data array with missing values filled.
    """
    isnan = np.isnan(data)
    nan_indices = np.where(isnan)[0]
    
    if len(nan_indices) == 0:
        return data  # No NaNs to fill

    gaps = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)
    not_nan_indices = np.where(~isnan)[0]
    not_nan_values = data[not_nan_indices]
    interpolator = PchipInterpolator(not_nan_indices, not_nan_values)
    
    for gap in gaps:
        if len(gap) > 0:
            gap_size = len(gap) + 2 if (gap[0] > 0 and gap[-1] < len(data) - 1) else len(gap) + 1
            if gap_size <= max_gap + 1:
                data[gap] = interpolator(gap)
    
    return data

def fill_data(matrix: np.ndarray, days: np.ndarray) -> np.ndarray:
    """
    Fill outliers and missing values in the data matrix.

    Parameters:
        matrix (np.ndarray): Input data matrix.
        days (np.ndarray): Array of day indices.
    
    Returns:
        np.ndarray: Filled data matrix.
    """
    start_index = np.where(np.diff(days) > 1)[0]
    start_index = np.concatenate(([0], start_index + 1, [len(days)]))
    
    comb_1d = np.array([])
    for i in range(len(start_index) - 1):
        matrix_1d = matrix[:, start_index[i]:start_index[i+1]].reshape(-1, order='F')
        matrix_1d = fill_outliers(matrix_1d)
        matrix_1d = fill_missing(matrix_1d)
        comb_1d = np.concatenate((comb_1d, matrix_1d))
    
    filled_data = comb_1d.reshape(matrix.shape, order='F')
    return filled_data

def extract_json(filename: str) -> pd.DataFrame:
    """
    Extract and process JSON data into a DataFrame.

    Parameters:
        filename (str): Path to the JSON file.
    
    Returns:
        pd.DataFrame: Processed data as a DataFrame.
    """
    js = read_json(filename)
    session_date = datetime.strptime(js['SessionDate'], '%Y-%m-%dT%H:%M:%SZ')
    lead_location = js['LeadConfiguration']['Final'][0]['LeadLocation'].split('.')[-1]
    print(f"{session_date} - {lead_location}")
    
    if 'LFPTrendLogs' in js['DiagnosticData']:
        data = js['DiagnosticData']['LFPTrendLogs']
        data_left, data_right = [], []
        
        if 'HemisphereLocationDef.Left' in data:
            for key in data['HemisphereLocationDef.Left']:
                data_left.extend(data['HemisphereLocationDef.Left'][key])
        
        if 'HemisphereLocationDef.Right' in data:
            for key in data['HemisphereLocationDef.Right']:
                data_right.extend(data['HemisphereLocationDef.Right'][key])
        
        date_time = sorted(list(set([item['DateTime'] for item in data_left + data_right])))
        LFP = np.full((2, len(date_time)), np.nan)
        stim_amp = np.full((2, len(date_time)), np.nan)
        
        left_indices = {item['DateTime']: idx for idx, item in enumerate(data_left)}
        for item in data_left:
            idx = date_time.index(item['DateTime'])
            LFP[0, idx] = item['LFP']
            stim_amp[0, idx] = item['AmplitudeInMilliAmps']
        
        right_indices = {item['DateTime']: idx for idx, item in enumerate(data_right)}
        for item in data_right:
            idx = date_time.index(item['DateTime'])
            LFP[1, idx] = item['LFP']
            stim_amp[1, idx] = item['AmplitudeInMilliAmps']
        
        date_time = pd.to_datetime(date_time, format='%Y-%m-%dT%H:%M:%SZ', utc=True)
        LFP_trend = pd.DataFrame({
            'Timestamp': date_time,
            'LFP Amp Left': LFP[0, :],
            'LFP Amp Right': LFP[1, :],
            'Stim Amp Left': stim_amp[0, :],
            'Stim Amp Right': stim_amp[1, :]
        })
    else:
        LFP_trend = pd.DataFrame()
    
    return LFP_trend

def read_json(filename: str) -> Dict:
    """
    Read a JSON file and return the data as a dictionary.

    Parameters:
        filename (str): Path to the JSON file.
    
    Returns:
        Dict: Data read from the JSON file.
    """
    with open(filename, 'r') as f:
        return json.load(f)

def cosinor(t: np.ndarray, y: np.ndarray, w: float, num_components: int, num_peaks: int) -> Tuple[np.ndarray, np.ndarray, float, Any]:
    """
    Perform cosinor analysis on the input data.
    
    Parameters:
        t (np.ndarray): Time values.
        y (np.ndarray): Signal values.
        w (float): Period of the cosinor analysis.
        num_components (int): Number of harmonic components.
        num_peaks (int): Number of peaks to identify.
    
    Returns:
        Tuple: Containing amplitude, acrophase, p-value, and fit model.
    """
    if len(t) < 4:
        raise ValueError('There must be at least four time measurements.')

    t = np.asarray(t).flatten()
    y = np.asarray(y).flatten()
    valid_idx = ~np.isnan(t)
    t = t[valid_idx]
    y = y[valid_idx]
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    y = y[sort_idx]
    w = w / 24  # convert period from hours to days

    X_fit = []
    for i in range(1, num_components + 1):
        A = np.sin(t / (w / i) * 2 * np.pi)
        B = np.cos(t / (w / i) * 2 * np.pi)
        X_fit.append(A)
        X_fit.append(B)
    
    X_fit = np.column_stack(X_fit)
    X_fit = add_constant(X_fit)
    fit_model = OLS(y, X_fit).fit()
    f = fit_model.fittedvalues
    mesor = np.median([np.min(f), np.max(f)])
    p_value = fit_model.f_pvalue

    try:
        peaks, _ = find_peaks(f, height=mesor, distance=1)
        peak_locs = t[peaks]
        disc_peaks = pd.cut(np.mod(peak_locs, 1), bins=num_peaks, labels=False)
        disc_acro = np.unique(disc_peaks)
    except Exception:
        disc_peaks = np.array([])
        disc_acro = np.array([])
    
    amplitude = []
    acrophase = []
    if len(disc_acro) > 1:
        for i in range(num_peaks):
            peak_group = disc_peaks == disc_acro[-i-1]
            acrophase.append(np.median(np.mod(peak_locs[peak_group], 1) * 2 * np.pi))
            amplitude.append((np.median(f[peaks][peak_group]) - np.min(f)) / 2)
    elif len(disc_acro) == 0:
        acrophase = np.nan
        amplitude = np.nan
    else:
        acrophase = np.mod(peak_locs[-1], 1) * 2 * np.pi
        amplitude = (f[peaks][-1] - np.min(f)) / 2
    
    return np.array(amplitude), np.array(acrophase), p_value, fit_model

def ensure_key_exists(data: Dict[str, Any], key: str, j: str, hemisphere: int) -> None:
    """
    Ensure that a key and its nested keys exist in the dictionary, initializing them if necessary.
    
    Parameters:
        data (Dict[str, Any]): The data dictionary.
        key (str): The key to ensure exists.
        j (Any): The sub-key to ensure exists under the key.
        hemisphere (Any): The sub-key to ensure exists under the sub-key j.
    """
    if key not in data:
        data[key] = {}
    if j not in data[key]:
        data[key][j] = {}
    if hemisphere not in data[key][j]:
        data[key][j][hemisphere] = None  # Initialize with None or an appropriate default value

def merge_advanced_data(advanced_data: Dict[str, Any], percept_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge advanced calculation data back into the percept data.

    Parameters:
        advanced_data (Dict[str, Any]): The advanced calculation data.
        percept_data (Dict[str, Any]): The percept data.

    Returns:
        Dict[str, Any]: The updated percept data with merged advanced data.
    """
    subject_list = percept_data['LFP_filled_matrix'].keys()
    model_list = ['cosinor', 'linearAR', 'nonlinear_AR']
    orig_model_list = ['Cosinor', 'LinAR', 'NN_AR']
    
    for model_idx in range(len(model_list)):
        dest_model = model_list[model_idx]
        orig_model = orig_model_list[model_idx]
        percept_data[f'{dest_model}_matrix'] = {}
        percept_data[f'{dest_model}_R2'] = {}
        
        if orig_model not in advanced_data:
            continue
        
        for subject in subject_list:
            if subject not in percept_data[f'{dest_model}_matrix']:
                percept_data[f'{dest_model}_matrix'][subject] = {}
            if subject not in percept_data[f'{dest_model}_R2']:
                percept_data[f'{dest_model}_R2'][subject] = {}
            
            for hemisphere in [0, 1]:
                percept_data[f'{dest_model}_matrix'][subject][hemisphere] = advanced_data[orig_model][hemisphere][f'{orig_model}_{subject}_Raw'].to_numpy()
                fill_r2_data(percept_data, advanced_data, dest_model, orig_model, subject, hemisphere)
                
    return percept_data

def fill_r2_data(percept_data: Dict[str, Any], advanced_data: Dict[str, Any], dest_model: str, orig_model: str, subject: Any, hemisphere: int) -> None:
    """
    Fill the R² data for the specified model, subject, and hemisphere.

    Parameters:
        percept_data (Dict[str, Any]): The percept data dictionary.
        advanced_data (Dict[str, Any]): The advanced calculation data dictionary.
        dest_model (str): The destination model name (e.g., 'cosinor', 'linearAR', 'nonlinear_AR').
        orig_model (str): The original model name (e.g., 'Cosinor', 'LinAR', 'NN_AR').
        subject (Any): The subject name.
        hemisphere (int): The hemisphere index (0 or 1).
    """
    try:
        if subject in percept_data['days']:
            subject_days = percept_data['days'][subject][hemisphere]
            metric_days = advanced_data[orig_model][hemisphere][f'{orig_model}_{subject}_Metric']['Day'].values
            
            common_days, subject_idx, metric_idx = np.intersect1d(subject_days, metric_days, return_indices=True)
            
            temp_data = np.full(len(subject_days), np.nan)
            temp_data[subject_idx] = advanced_data[orig_model][hemisphere][f'{orig_model}_{subject}_Metric'].iloc[metric_idx]['R2'].to_numpy()
            
            percept_data[f'{dest_model}_R2'][subject][hemisphere] = temp_data
    except KeyError:
        pass

