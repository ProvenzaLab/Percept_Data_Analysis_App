# Percept Data Analysis Guide

## Overview

The Percept Data Analysis App helps users process and analyze Medtronic Percept BrainSense Timeline data efficiently. This guide explains the app's main parameters, methods, and usage.

## Adjustable Parameters

**Patient Input** : Add individual patients into the app database
- There are 3 required parameters
    
    1. Patient ID: Unique patient identifier
    2. Initial DBS Programming: The date DBS stimulation was initially turned on. Must be in YYYY-MM-DD format (2000-01-31).
    3. Response Status: The response status of the DBS patient.
        
        - Response Date: If the patient is a responder, enter the date that clinical response was achieved. Enter the date in YYYY-MM-DD format or simply the number of days after initial DBS programming that response was achieved.

- Once the patient's parameters are entered, a directory selection window will appear. Select the parent directory containing all of the corresponding patient's JSON files. 

**Model Parameters** : Adjust model parameters in the settings menu

- Artifact Correction Method

    - High-amplitude artifacts occur when voltage readings exceed the Medtronic Percept device's maximum sensing capabilities. The app includes 3 different correction methods that can be used. For more detailed information about these overvoltage events, refer [here](https://www.medrxiv.org/content/10.1101/2025.07.23.25331987v1).

    1. Simple Threshold: Remove all values with at least one overvoltage event
    2. Threshold and Interpolation: Identify all values with at least one overvoltage event. Remove them and interpolate missing data using PCHIP interpolation.
    3. Overvoltage Event Removal: Identify and recalculate all values with at least one overvoltage event. Each 10-minute interval's amplitude average is calculated without the overvoltage events, preserving non-artifact neural data. This method is preferred.

- Window Size
    
    - The sliding window size (*n*) that the autoregressive model will train and test on. The autoregressive model makes predictions on the neural LFP values by training on the first *n*-1 days' values and predicting the last day's values. The *n*-day window slides across the patient's entire recording duration with a stride of one day. The default window size is 3 days.

- Delta Normalization

    - Normalize a patient's R² values to their pre-DBS baseline average. The mean of the patient's pre-DBS R² values is subtracted from all R² values. If no pre-DBS data is available for the patient, the unnormalized R² values are used.

- AR(k) model

    - Option to use an AR(k) model instead of the default AR(1) model. If the AR(k) model is selected, the number of lag terms used by the AR(k) model can be specified. The default number of lag terms is 72, up to a half day's worth of neural data.



## Methods

### 1. Generate Raw Data

Read all JSON files within the patient's provided directory. All raw chronic LFP data from all patients is stored in a pandas DataFrame. Parameter changes to amplitude, pulse width, frequency, stimulation contacts, and sensing frequency for all patients are stored in another pandas DataFrame. 

### 2. Normalize and Correct Data

Process all raw LFP data. Remove duplicate data readings, interpolate outliers and missing rows, correct overvoltage events using the specified method, and per-day z-score LFP data. The lag terms are also calculated at this step.

### 3. Model Data

An autoregressive (AR) model is used to predict the LFP data. The model is applied to each patient's neural data with a sliding window of size *n* and a 1-day stride. The model can be entirely causal by training on the previous *n*-1 days' of neural data and predicting the window's last day of neural data. The model can also be applied as a non-causal method where the model is trained on neural data from the days flanking the window's center day and predicting that center day's neural data. 

If an AR(k) model is used, significant lag terms are determined using an *n*-fold cross-validation with an ordinary least squares regression model. For each fold of the cross-validation, significant lag terms (*p* < 0.05) were recorded. If the lag term was significant for more than half the folds, then the lag term was deemed significant. The default number of folds is 5. Lag terms with more than 50% `nan` values were dropped. 

The R² of the z-scored LFP data and the AR predicted data is calculated. 

### 4. Visualization

A 3-panel plot is displayed after the analysis pipeline finishes. The patients and hemisphere can be adjusted with the dropdown menus in the top left corner of the window. General patient data and sample information are displayed in the left panel. The top plot shows the z-scored LFP data (gray) and the AR predicted LFP data (green) over time (CT time). The bottom left plot shows a scatter of the R² values over time (days since DBS activation) colored by the clinical status. A 5-day moving average of the R² values is also shown in gray. The bottom right plot shows a violin plot of the R² values separated by clinical state.

Stimulation parameter changes can be displayed on the LFP time series plot as well. Hovering over specific parameter change lines will display the specific parameter changes at that time. The raw LFP data, R² values, and plot can saved as well. 

## Usage Steps

1. Save patients in the app.
2. Configure analysis settings.
3. Click "Run Analysis" to process data.
4. View results and download output.

## Other Tools
- Refer to the README for advanced configuration options and developer tools.
- The `terminal_runner.py` file contains a demo of the analysis pipeline

## Troubleshooting

- If analysis fails, check file format and parameter settings.
- For large datasets, processing may take longer.

If problems persist, email [jz186@rice.edu](mailto:jz186@rice.edu) with a screenshot and description of the issue.
