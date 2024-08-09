"""
generate_data.py

This script processes neural data for a given subject, handling raw data,
normalizing it, and organizing it into matrices based on timestamps. The
final data is stored in a nested dictionary structure with specific fields.

Functions:
    generate_data(subject_name, percept_data=None, zone_index=None, time_zone='America/Chicago')

Usage:
    Run the script directly to generate data for a subject:
    ```
    if __name__ == "__main__":
        percept_data = None
        zone_index = None
        percept_data, zone_index = generate_data(subject_name='009', percept_data=percept_data, zone_index=zone_index)
        print(percept_data)
        print(zone_index)
    ```
"""

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import data_processing.utils as utils
from datetime import datetime
import pytz

def generate_data(subject_name, percept_data=None, zone_index=None, time_zone='America/Chicago', file_list = None):
    
    """
    Process percept-medtronic neural data for a given subject and organize it into a nested dictionary structure.

    Parameters:
        subject_name (str): The ID of the subject whose data will be added (e.g., "B001").
        percept_data (dict, optional): Existing percept data dictionary to add this patient data to. Defaults to None.
        zone_index (dict, optional): Existing zone index dictionary to add this patient data to. Defaults to None.
        time_zone (str, optional): Time zone for the data. Defaults to 'America/Chicago'.

    Returns:
        dict: Updated percept_data nested dictionary.
        dict: Updated zone_index dictionary.
        
    VISIT https://github.com/shethlab/PerceptDataAnalysis/blob/main/CircadianPaper/generate_data.m 
    For more detailed documentation 
    """
    # Choosing files and DBS onset date
    if file_list is None:
        file_list = filedialog.askopenfilenames(title='Select patient JSON files', filetypes=[('JSON files', '*.json')])
    
    time_zone = pytz.timezone(time_zone)
    
    input_args = utils.read_json("param.json")
    DBS_onset_str = input_args['dbs_date']
    DBS_onset = datetime.strptime(DBS_onset_str, '%m-%d-%Y')
    DBS_onset = time_zone.localize(DBS_onset)

    # Raw data, normalized data, and timestamps
    if percept_data and subject_name in percept_data.get('raw_data', {}):
        raw_data = percept_data['raw_data'][subject_name]
    else:
        raw_data = pd.DataFrame()

    # Import and concatenate data from JSONs
    for filepath in file_list:
        raw = utils.extract_json(filepath)
        if not raw.empty:
            raw_data = pd.concat([raw_data, raw], ignore_index=True)

    if 'Timestamp' not in raw_data.columns:
        raise KeyError('Timestamp column is missing in the raw data')

    raw_data['Timestamp'] = pd.to_datetime(raw_data['Timestamp']).dt.tz_convert(time_zone)

    # Step 1: Remove duplicate data points
    raw_data = raw_data.drop_duplicates(subset=['Timestamp'])
    unique_dates = raw_data['Timestamp']
    
    # Step 2: Discretize data timepoints into 10 minute time-of-day (TOD) bins without date
    # Generate 10-minute bins between 0 to 1500 minutes (25 hours)
    time_hist = np.arange(0, 25*60 + 1, 10)
    # Assuming unique_dates is a list of datetime strings or datetime objects
    unique_dates = pd.to_datetime(unique_dates)
    # Convert unique_dates to total minutes of the day
    time_of_day_minutes = (pd.to_datetime(unique_dates) - pd.to_datetime(unique_dates).dt.floor('D')).view('int64') / (60 * 1e9)

    # Discretize time of day into bins
    disc_TOD = np.digitize(time_of_day_minutes, time_hist)
    # Shift daylight savings time by 24 hours to make it <24 hours
    disc_TOD[disc_TOD > 24*6] = disc_TOD[disc_TOD > 24*6] - 24*6
    
    # Step 3: Discretize data timepoints into dates without times
    rounded_dates = unique_dates.dt.floor('d')
    unique_rounded_dates = np.unique(rounded_dates)

    # Step 4: Initialize matrices
    LFP_matrix = {1: np.full((144, len(unique_rounded_dates)), np.nan), 2: np.full((144, len(unique_rounded_dates)), np.nan)}
    stim_matrix = {1: np.full((144, len(unique_rounded_dates)), np.nan), 2: np.full((144, len(unique_rounded_dates)), np.nan)}
    time_matrix = {1: np.full((144, len(unique_rounded_dates)), np.nan, dtype='datetime64[ns]'), 2: np.full((144, len(unique_rounded_dates)), np.nan, dtype='datetime64[ns]')}


    disc_TOD_radix  = disc_TOD - 1
    # Step 5: Populate matrices
    for i, unique_date in enumerate(unique_rounded_dates):
        idx = np.where(np.isin(rounded_dates, unique_date))[0]
        
        LFP_matrix[1][disc_TOD_radix[idx], i] = raw_data.iloc[idx]['LFP Amp Left'].values
        time_matrix[1][disc_TOD_radix[idx], i] = unique_dates.iloc[idx].values
        stim_matrix[1][disc_TOD_radix[idx], i] = raw_data.iloc[idx]['Stim Amp Left'].values

        LFP_matrix[2][disc_TOD_radix[idx], i] = raw_data.iloc[idx]['LFP Amp Right'].values
        time_matrix[2][disc_TOD_radix[idx], i] = unique_dates.iloc[idx].values
        stim_matrix[2][disc_TOD_radix[idx], i] = raw_data.iloc[idx]['Stim Amp Right'].values
    
    # Step 6: Generate per-hemisphere data cells
    DBS_time = {}
    LFP_norm_matrix = {}
    LFP_filled_matrix = {}
    for hemisphere in [1, 2]:

        all_nan_days = np.all(np.isnan(LFP_matrix[hemisphere]), axis=0)

        # List of dates containing data relative to DBS onset
        unique_cal_days = unique_rounded_dates[~all_nan_days]

        # Ensure all dates are in Chicago time for calculations
        DBS_onset_naive = pd.to_datetime(DBS_onset)
        unique_cal_days_naive = pd.to_datetime(unique_cal_days)

        # Flatten the time_matrix for the hemisphere and convert to Chicago time
        time_matrix_naive_flat = pd.to_datetime(time_matrix[hemisphere].ravel()).to_numpy()

        # Calculate DBS_time for unique_cal_days
        DBS_time[hemisphere] = np.round((unique_cal_days_naive - DBS_onset_naive).total_seconds() / 86400).to_numpy()

        # Calculate time_matrix relative to DBS_onset and reshape it back to the original shape
        time_matrix_relative = (time_matrix_naive_flat - np.datetime64(DBS_onset_naive)) / np.timedelta64(1, 'D')

        # Replace NaT values with np.nan
        time_matrix_relative[np.isnat(time_matrix_naive_flat)] = np.nan
        # Reshape the array to the original shape and update the time_matrix
        time_matrix[hemisphere] = time_matrix_relative.reshape(time_matrix[hemisphere].shape)
                
        # Remove empty days from LFP data matrix and create nan-filled, outlier-removed, per-day normalized matrix
        LFP_matrix[hemisphere] = LFP_matrix[hemisphere][:, ~all_nan_days]
        time_matrix[hemisphere] = time_matrix[hemisphere][:, ~all_nan_days]
        stim_matrix[hemisphere] = stim_matrix[hemisphere][:, ~all_nan_days]
        LFP_norm_matrix[hemisphere] = (LFP_matrix[hemisphere] - np.nanmean(LFP_matrix[hemisphere], axis=0)) / np.nanstd(LFP_matrix[hemisphere], axis=0, ddof=1) # using sample std formula(n-1)
        filled = utils.fill_data(LFP_matrix[hemisphere], DBS_time[hemisphere])  # Assuming fill_data function is defined
        LFP_filled_matrix[hemisphere] = (filled - np.nanmean(filled, axis=0)) / np.nanstd(filled, axis=0, ddof=1)

    if percept_data is None:
        percept_data = {
            'raw_data': {},
            'days': {},
            'time_matrix': {},
            'LFP_raw_matrix': {},
            'LFP_norm_matrix': {},
            'LFP_filled_matrix': {},
            'stim_matrix': {}
        }
    
    if zone_index is None:
        zone_index = {
            'subject': [],
            'responder': [],
            'non_responder': [],
            'hypomania': []
        }

    # Update zone_index
    if subject_name in zone_index['subject']:
        subject_idx = zone_index['subject'].index(subject_name)
        if input_args.get('zone_idx') and input_args['zone_idx'].lower() == 'y':
            zone_index['responder'][subject_idx] = input_args['responder_zone_idx']
            zone_index['non_responder'][subject_idx] = input_args['non_responder_idx']
            zone_index['hypomania'][subject_idx] = ''
    else:
        zone_index['subject'].append(subject_name)
        zone_index['responder'].append(input_args.get('responder_zone_idx'))
        zone_index['non_responder'].append(input_args.get('non_responder_idx'))
        zone_index['hypomania'].append(input_args.get('hypomania'))

    # Update percept_data
    if subject_name in percept_data['raw_data']:
        subject_idx = percept_data['raw_data'].index(subject_name)
    else:
        subject_idx = len(percept_data['raw_data'])
        percept_data['raw_data'][subject_name] = raw_data
        percept_data['days'][subject_name] = [None, None]
        percept_data['time_matrix'][subject_name] = [None, None]
        percept_data['LFP_raw_matrix'][subject_name] = [None, None]
        percept_data['LFP_norm_matrix'][subject_name] = [None, None]
        percept_data['LFP_filled_matrix'][subject_name] = [None, None]
        percept_data['stim_matrix'][subject_name] = [None, None]

    percept_data['days'][subject_name][0] = DBS_time[1]
    percept_data['days'][subject_name][1] = DBS_time[2]
    percept_data['time_matrix'][subject_name][0] = time_matrix[1]
    percept_data['time_matrix'][subject_name][1] = time_matrix[2]
    percept_data['LFP_raw_matrix'][subject_name][0] = LFP_matrix[1]
    percept_data['LFP_raw_matrix'][subject_name][1] = LFP_matrix[2]
    percept_data['LFP_norm_matrix'][subject_name][0] = LFP_norm_matrix[1]
    percept_data['LFP_norm_matrix'][subject_name][1] = LFP_norm_matrix[2]
    percept_data['LFP_filled_matrix'][subject_name][0] = LFP_filled_matrix[1]
    percept_data['LFP_filled_matrix'][subject_name][1] = LFP_filled_matrix[2]
    percept_data['stim_matrix'][subject_name][0] = stim_matrix[1]
    percept_data['stim_matrix'][subject_name][1] = stim_matrix[2]

    return percept_data, zone_index

