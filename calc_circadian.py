import numpy as np
import utils
from EntropyHub import SampEn
import calc_circadian_advanced

def calc_circadian(percept_data, zone_index, cosinor_window_left=2, cosinor_window_right=2, include_nonlinear=0):
    """
    Calculate various circadian metrics including cosinor R2, amplitude, acrophase, linear AR R2, 
    nonlinear AR R2 (optional), and sample entropy for Percept data.

    Parameters:
        percept_data (dict): Data structure containing LFP (Local Field Potential) and time-series data.
        zone_index (dict): Structure containing clinical response days (response, non-response, hypomania).
        cosinor_window_left (int): Number of days prior to the day of interest for cosinor calculation.
        cosinor_window_right (int): Number of days after the day of interest for cosinor calculation.
        include_nonlinear (int, optional): Flag to include nonlinear AR model calculations. Defaults to 0.

    Returns:
        dict: Updated percept_data structure with calculated metrics.
    """
    
    # Validate cosinor window inputs
    cosinor_window = [cosinor_window_left, cosinor_window_right]
    if not all(isinstance(x, int) and x >= 0 for x in cosinor_window):
        raise ValueError('Cosinor window inputs must be integers >= 0.')

    # Determine models to include based on the include_nonlinear flag
    models = ['Cosinor', 'LinAR', 'SE'] if include_nonlinear != 1 else ['Cosinor', 'LinAR', 'NN_AR', 'SE']

    for subject in percept_data['LFP_norm_matrix'].keys():
        num_components = 1
        num_peaks = 1

        # Validate cosinor inputs
        if not all(isinstance(x, int) and x >= 1 for x in [num_components, num_peaks]):
            raise ValueError('Cosinor inputs must be positive integers.')

        for hemisphere in [0, 1]:
            days = percept_data['days'][subject][hemisphere]
            LFP_norm = percept_data['LFP_norm_matrix'][subject][hemisphere]
            LFP_filled = percept_data['LFP_filled_matrix'][subject][hemisphere]
            time_matrix = percept_data['time_matrix'][subject][hemisphere]

            # Skip processing if there is a size mismatch between day values and LFP data
            if len(days) != LFP_filled.shape[1]:
                continue

            # Identify indices of discontinuous days
            start_index = np.where(np.diff(days) > 1)[0]
            start_index = np.concatenate(([0], start_index + 1, [len(days)])) if len(start_index) else np.array([0, len(days)])

            # Initialize arrays for metrics
            sample_entropy = np.full(len(days), np.nan)
            acro = np.full((len(days), num_peaks), np.nan)
            amp = np.full((len(days), num_peaks), np.nan)
            p_values = np.full(len(days), np.nan)

            # Calculate cosinor metrics
            for i in range(len(days)):
                if any((start_index > i - cosinor_window_left) & (start_index <= i + cosinor_window_right)) or len(days) < i + cosinor_window_right:
                    continue
                y = LFP_norm[:, (i - cosinor_window_left):(i + cosinor_window_right + 1)].T.flatten()
                t = time_matrix[:, (i - cosinor_window_left):(i + cosinor_window_right + 1)].T.flatten()
                amp[i, :num_peaks], acro[i, :num_peaks], p_values[i], _ = utils.cosinor(t, y, 24, num_components, num_peaks)

            # Template acrophase and p-value calculation for plotting
            template_acro = np.full((len(days), num_peaks), np.nan)
            template_p = np.full(len(days), np.nan)
            for i in range(len(days)):
                print(f'{subject} - {i + 1}') # Used for logging disregard if not needed
                if any((start_index > i) & (start_index <= i)) or len(days) < i:
                    continue
                y = LFP_norm[:, i].T.flatten()
                t = time_matrix[:, i].T.flatten()
                _, template_acro[i, :num_peaks], template_p[i], _ = utils.cosinor(t, y, 24, num_components, num_peaks)

            # Calculate sample entropy
            for i in range(len(days)):
                print(f'{subject} - {i + 1}') # Used for logging disregard if not needed
                if any((start_index > i) & (start_index <= i)) or len(days) < i + 1:
                    continue
                y_filled = LFP_filled[:, i].T.flatten()
                s = SampEn(y_filled, m=2, tau=1, r=3.6)
                sample_entropy[i] = s[0][2]  # Adjust indexing based on the structure of 's'

            # Save the calculated metrics to percept_data
            keys = ['entropy', 'amplitude', 'acrophase', 'cosinor_p', 'template_acro', 'template_p']
            for key in keys:
                utils.ensure_key_exists(percept_data, key, subject, hemisphere)
            percept_data['entropy'][subject][hemisphere] = sample_entropy
            percept_data['amplitude'][subject][hemisphere] = amp
            percept_data['acrophase'][subject][hemisphere] = acro
            percept_data['cosinor_p'][subject][hemisphere] = p_values
            percept_data['template_acro'][subject][hemisphere] = template_acro
            percept_data['template_p'][subject][hemisphere] = template_p

    # Perform advanced circadian calculations
    mat_file = [percept_data, zone_index]
    all_components = [num_components]

    advanced_calc_data = {}
    for model in models:
        advanced_calc_data[model] = {}
        for hemisphere in [0, 1]:
            advanced_calc_data[model][hemisphere] = calc_circadian_advanced.main(
                hemi=hemisphere, 
                mat_file=mat_file, 
                components=all_components, 
                pt_index=[0], 
                pt_names=zone_index['subject'], 
                models=[model]
            )
   
    # Merge advanced calculation results back into percept_data
    percept_data = utils.merge_advanced_data(percept_data=percept_data, advanced_data=advanced_calc_data)
    
    return percept_data
