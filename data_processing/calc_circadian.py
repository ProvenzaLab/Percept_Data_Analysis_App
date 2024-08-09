

import numpy as np
import pandas as pd
import utils
import time
from EntropyHub import SampEn
import calc_circadian_advanced

def calc_circadian(percept_data, zone_index, cosinor_window_left=2, cosinor_window_right=2, include_nonlinear=0):
    """
    Calculate various data metrics including cosinor R2, amplitude, acrophase, linear AR R2, nonlinear AR R2, and sample entropy.

    Parameters:
        percept_data (dict): The data structure containing the Percept data.
        zone_index (dict): The structure containing the list of days in which patients are behaviorally noted as being in clinical response, non-response, or hypomania.
        cosinor_window_left (int): The number of days prior to the day of interest to include in the calculation window for cosinor.
        cosinor_window_right (int): The number of days after the day of interest to include in the calculation window for cosinor.
        include_nonlinear (int, optional): A flag which, when set to 1, includes calculations for the nonlinear autoregressive model. Defaults to 0.
        is_demo (int, optional): A flag which, when set to 1, signals that the demo dataset is being run. Defaults to 0.

    Returns:
        dict: The updated percept_data structure including all of the input information, as well as the new calculated data.
    """

    # Warning if improper window size inputs
    cosinor_window = [cosinor_window_left, cosinor_window_right]
    if not all(isinstance(x, int) and x >= 0 for x in cosinor_window):
        raise ValueError('Cosinor window inputs must be integers >= 0.')

    # Checking if nonlinear AR is to be included in model list
    if include_nonlinear != 1:
        print('Models to analyze: cosinor, linear AR, sample entropy.')
        models = ['Cosinor', 'LinAR', 'SE']
    else:
        print('Models to analyze: cosinor, linear AR, nonlinear AR, sample entropy.')
        models = ['Cosinor', 'LinAR', 'NN_AR', 'SE']



    time.sleep(2)

    # Read command line arguments from JSON
    input_struct = utils.read_json('param.json')

    for j in percept_data['LFP_norm_matrix'].keys():

        num_components = 1
        num_peaks = 1

        # Warning if improper cosinor inputs
        if not all(isinstance(x, int) and x >= 1 for x in [num_components, num_peaks]):
            raise ValueError('Cosinor inputs must be positive integers.')

        for hemisphere in [0, 1]:
            days = percept_data['days'][j][hemisphere]
            LFP_norm = percept_data['LFP_norm_matrix'][j][hemisphere]
            LFP_filled = percept_data['LFP_filled_matrix'][j][hemisphere]
            time_matrix = percept_data['time_matrix'][j][hemisphere]

            # Check that the day values line up with the data and skip if not
            if len(days) != LFP_filled.shape[1]:
                print('Size mismatch between day values and LFP data. Skipping this hemisphere.')
                continue

            # Find indices of discontinuous days of data
            start_index = np.where(np.diff(days) > 1)[0]
            try:
                start_index = np.concatenate(([0], start_index + 1, [len(days)]))
            except:
                start_index = np.array([0, len(days)])

            # Initializing metrics
            sample_entropy = np.full(len(days), np.nan)
            acro = np.full((len(days), num_peaks), np.nan)
            amp = np.full((len(days), num_peaks), np.nan)
            p_values = np.full(len(days), np.nan)

            for i in range(len(days)):
                print(f'{j} - {i + 1}')
                if any((start_index > i - cosinor_window_left) & (start_index <= i + cosinor_window_right)) or len(days) < i + cosinor_window_right:
                    # Skipping calculations if there are full-day or greater gaps in data in the specified window
                    continue
                y = LFP_norm[:, (i - cosinor_window_left):(i + cosinor_window_right + 1)].T.flatten()
                t = time_matrix[:, (i - cosinor_window_left):(i + cosinor_window_right + 1)].T.flatten()
                
                # Calculation of cosinor amplitude, acrophase, p-value
                amp[i, :num_peaks], acro[i, :num_peaks], p_values[i], _ = utils.cosinor(t, y, 24, num_components, num_peaks)

            # Hard-coded 1-day window to generate daily acrophases for template-plotting code
            template_acro = np.full((len(days), num_peaks), np.nan)
            template_p = np.full(len(days), np.nan)
            for i in range(len(days)):
                print(f'{j} - {i + 1}')
                if any((start_index > i) & (start_index <= i)) or len(days) < i:
                    # Skipping calculations if there are full-day or greater gaps in data in the specified window
                    continue
                y = LFP_norm[:, i].T.flatten()
                t = time_matrix[:, i].T.flatten()
                
                # Calculation of cosinor amplitude, acrophase, p-value
                _, template_acro[i, :num_peaks], template_p[i], _ = utils.cosinor(t, y, 24, num_components, num_peaks)

            for i in range(len(days)):
                print(f'{j} - {i + 1}')
                if any((start_index > i) & (start_index <= i)) or len(days) < i + 1:
                    # Skipping calculations if there are full-day or greater gaps in data in the specified window
                    continue
                y_filled = LFP_filled[:, i].T.flatten()
                
                # Calculation of sample entropy
                s = utils.custom_SampEn(y_filled, m=2, tau=1, r=3.6)
                sample_entropy[i] = s[2]  # Adjust indexing based on the structure of 's'

            # Saving the patient/hemisphere metrics to the overall data structure
            keys = ['entropy', 'amplitude', 'acrophase', 'cosinor_p', 'template_acro', 'template_p']
            for key in keys:
                utils.ensure_key_exists(percept_data, key, j, hemisphere)
            percept_data['entropy'][j][hemisphere] = sample_entropy
            percept_data['amplitude'][j][hemisphere] = amp
            percept_data['acrophase'][j][hemisphere] = acro
            percept_data['cosinor_p'][j][hemisphere] = p_values
            percept_data['template_acro'][j][hemisphere] = template_acro
            percept_data['template_p'][j][hemisphere] = template_p

    # Save data to temporary JSON file to pass to Python for advanced calculations

    # Run the python file calc_circadian_advanced
    mat_file=[percept_data, zone_index]
    all_components = [num_components]
    
    advanced_calc_data = {}
    for model in models:
        if model not in advanced_calc_data:
            advanced_calc_data[model] = {}  # Initialize the nested dictionary for the model
        for hemisphere in [0, 1]:
            print(f'Running - {model} Hemisphere {hemisphere}')
            advanced_calc_data[model][hemisphere] = calc_circadian_advanced.main(
                hemi=hemisphere, 
                mat_file=mat_file, 
                components=all_components, 
                pt_index=[0], 
                pt_names=zone_index['subject'], 
                models=[model]
            )
   
    percept_data = utils.merge_advanced_data(percept_data=percept_data, advanced_data=advanced_calc_data)
    
    return percept_data
            

            
    
       