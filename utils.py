import numpy as np
from scipy.interpolate import PchipInterpolator
import json
import pandas as pd
from datetime import datetime
from scipy.signal import find_peaks
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
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
    # Create a mask for NaNs to ignore them during outlier processing
    not_nan = ~np.isnan(data)
    
    # Calculate median and scaled MAD for non-NaN values
    median = np.median(data[not_nan])
    mad = np.median(np.abs(data[not_nan] - median)) * 1.4826  # Scale MAD to match MATLAB
    
    # Define the threshold for outliers
    threshold = threshold_factor * mad
    
    # Identify outliers (ignoring NaNs)
    outliers = (np.abs(data - median) > threshold) & not_nan
    
    # Indices of non-outliers for interpolation
    valid_indices = np.where(not_nan & ~outliers)[0]
    valid_values = data[valid_indices]
    
    # Interpolator for PCHIP
    interpolator = PchipInterpolator(valid_indices, valid_values)
    
    # Replace outliers with interpolated values
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
    # Create a mask for NaNs
    isnan = np.isnan(data)
    
    # Find the indices of NaNs
    nan_indices = np.where(isnan)[0]
    
    if len(nan_indices) == 0:
        return data  # No NaNs to fill

    # Group the NaN indices into contiguous gaps
    gaps = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)
    
    # Interpolator for PCHIP
    not_nan_indices = np.where(~isnan)[0]
    not_nan_values = data[not_nan_indices]
    interpolator = PchipInterpolator(not_nan_indices, not_nan_values)
    
    # Fill NaNs for gaps of size <= max_gap
    for gap in gaps:
        if len(gap) > 0:
            # Compute the gap size relative to sample points
            if gap[0] > 0 and gap[-1] < len(data) - 1:
                # Gap is surrounded by non-missing values
                gap_size = gap[-1] - gap[0] + 1
                surrounding_gap_size = gap_size + 2  # Including the surrounding points
            else:
                # Gap is at the beginning or end
                gap_size = len(gap)
                surrounding_gap_size = len(gap) + 1
            
            if surrounding_gap_size <= max_gap + 1:
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
    # Extract json
    js = read_json(filename)
    
    # Display time of session and location of DBS lead to user
    session_date = datetime.strptime(js['SessionDate'], '%Y-%m-%dT%H:%M:%SZ')
    lead_location = js['LeadConfiguration']['Final'][0]['LeadLocation'].split('.')[-1]
    print(f"{session_date} - {lead_location}")
    
    if 'LFPTrendLogs' in js['DiagnosticData']:
        # Initialize data-holding variables
        data = js['DiagnosticData']['LFPTrendLogs']
        data_left = []
        data_right = []
        
        # Concatenate left hemisphere data
        if 'HemisphereLocationDef.Left' in data:
            for key in data['HemisphereLocationDef.Left']:
                data_left.extend(data['HemisphereLocationDef.Left'][key])
        
        # Concatenate right hemisphere data
        if 'HemisphereLocationDef.Right' in data:
            for key in data['HemisphereLocationDef.Right']:
                data_right.extend(data['HemisphereLocationDef.Right'][key])
        
        # Generate ascending list of unique datetimes
        date_time = sorted(list(set([item['DateTime'] for item in data_left + data_right])))
        
        # Create 2 row LFP/stim matrix with values sorted to match respective datetimes above
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
        
        # Export values
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

    # Transpose data into columns if needed
    t = np.asarray(t).flatten()
    y = np.asarray(y).flatten()

    # Remove nans and sort data by increasing time
    valid_idx = ~np.isnan(t)
    t = t[valid_idx]
    y = y[valid_idx]
    
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    y = y[sort_idx]

    w = w / 24  # convert period from hours to days

    # Generate sinusoidal component inputs for regression model
    X_fit = []
    for i in range(1, num_components + 1):
        A = np.sin(t / (w / i) * 2 * np.pi)
        B = np.cos(t / (w / i) * 2 * np.pi)
        X_fit.append(A)
        X_fit.append(B)
    
    X_fit = np.column_stack(X_fit)
    X_fit = add_constant(X_fit)  # add constant term
    
    # Fit the linear regression model to the sinusoidal inputs
    fit_model = OLS(y, X_fit).fit()
    
    # Calculation of miscellaneous measures
    f = fit_model.fittedvalues  # Raw data points of sinusoidal fit model
    mesor = np.median([np.min(f), np.max(f)])  # MESOR is defined as the cycle median
    p_value = fit_model.f_pvalue  # P value of model fit vs constant model
    
    # Acrophase and amplitude calculation
    try:
        # Use a distance of 1 to ensure peaks are at least one sample apart
        peaks, _ = find_peaks(f, height=mesor, distance=1)
        peak_locs = t[peaks]
        disc_peaks = pd.cut(np.mod(peak_locs, 1), bins=num_peaks, labels=False)
        disc_acro = np.unique(disc_peaks)
    except Exception as e:
        disc_peaks = np.array([])
        disc_acro = np.array([])
    
    amplitude = []
    acrophase = []
    if len(disc_acro) > 1:  # Multiple peaks per cycle
        for i in range(num_peaks):
            peak_group = disc_peaks == disc_acro[-i-1]
            acrophase.append(np.median(np.mod(peak_locs[peak_group], 1) * 2 * np.pi))  # outputting as radians
            amplitude.append((np.median(f[peaks][peak_group]) - np.min(f)) / 2)
    elif len(disc_acro) == 0:  # No peaks identified
        acrophase = np.nan
        amplitude = np.nan
    else:  # Single peak per cycle
        acrophase = np.mod(peak_locs[-1], 1) * 2 * np.pi  # outputting as radians
        amplitude = (f[peaks][-1] - np.min(f)) / 2
    
    return np.array(amplitude), np.array(acrophase), p_value, fit_model

def custom_SampEn(Sig: np.ndarray, m: int = 2, tau: int = 1, r: float = None, Logx: float = np.exp(1)) -> np.ndarray:
    """
    Calculate the sample entropy of a signal.

    Parameters:
        Sig (np.ndarray): Input signal.
        m (int, optional): Embedding dimension. Default is 2.
        tau (int, optional): Time delay. Default is 1.
        r (float, optional): Radius distance threshold. Default is 0.2 * SD(Sig).
        Logx (float, optional): Logarithm base. Default is np.exp(1).
    
    Returns:
        np.ndarray: Sample entropy values.
    """
    # Ensure Sig is a 1D numpy array
    Sig = np.squeeze(Sig)
    if Sig.ndim != 1:
        raise ValueError("Sig must be a one-dimensional array.")
    
    # Handle NaNs by removing them from the signal
    Sig = Sig[~np.isnan(Sig)]
    N = len(Sig)
    if N <= 10:
        raise ValueError("Sig must have more than 10 non-NaN elements.")
    
    # Set default value for r if not provided
    if r is None:
        r = 0.2 * np.std(Sig, ddof=1)
    
    # Initialize variables
    Counter = (abs(np.expand_dims(Sig, axis=1) - np.expand_dims(Sig, axis=0)) <= r) * np.triu(np.ones((N, N)), 1)
    M = np.hstack((m * np.ones(N - m * tau), np.repeat(np.arange(m - 1, 0, -1), tau)))
    A = np.zeros(m + 1)
    B = np.zeros(m + 1)
    A[0] = np.sum(Counter)
    B[0] = N * (N - 1) / 2

    for n in range(N - tau):
        ix = np.where(Counter[n, :] == 1)[0]
        for k in range(1, int(M[n]) + 1):
            ix = ix[ix + (k * tau) < N]
            p1 = np.tile(Sig[n:n + 1 + (tau * k):tau], (len(ix), 1))
            p2 = Sig[np.expand_dims(ix, axis=1) + np.arange(0, (k * tau) + 1, tau)]
            ix = ix[np.max(abs(p1 - p2), axis=1) <= r]
            if len(ix) > 0:
                Counter[n, ix] += 1
            else:
                break

    for k in range(1, m + 1):
        A[k] = np.sum(Counter > k)
        B[k] = np.sum(Counter[:, :-(k * tau)] >= k)

    with np.errstate(divide='ignore', invalid='ignore'):
        Samp = -np.log(A / B) / np.log(Logx)
    
    return Samp

def ensure_key_exists(data: Dict[str, Any], key: str, j: Any, hemisphere: Any) -> None:
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


def merge_advanced_data(advanced_data, percept_data):
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
                
                
def fill_r2_data(percept_data, advanced_data, dest_model, orig_model, subject, hemisphere):
    """
    Fills the R2 data for the specified model, subject, and hemisphere.

    Parameters:
    - percept_data: dictionary containing the percept data.
    - matlab_data: dictionary containing the matlab data.
    - model: the model name (e.g., 'cosinor', 'linearAR', 'NN_AR').
    - subject: the subject name.
    - hemisphere: the hemisphere index (0 or 1).
    """
    try:
        if subject in percept_data['days']:
            subject_days = percept_data['days'][subject][hemisphere]
            metric_days = advanced_data[orig_model][hemisphere][f'{orig_model}_{subject}_Metric']['Day'].values
            
            # Find common days
            common_days, subject_idx, metric_idx = np.intersect1d(subject_days, metric_days, return_indices=True)
            
            temp_data = np.full(len(subject_days), np.nan)
            temp_data[subject_idx] = advanced_data[orig_model][hemisphere][f'{orig_model}_{subject}_Metric'].iloc[metric_idx]['R2'].to_numpy()
            
            percept_data[f'{dest_model}_R2'][subject][hemisphere] = temp_data
    except KeyError:
        pass
    
    
def save_to_json(dictionary, filename):
    """
    Writes a dictionary to a JSON file.

    Parameters:
    dictionary (dict): The dictionary to write to the JSON file.
    filename (str): The name of the JSON file to write to.
    """
    
    with open(filename, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)