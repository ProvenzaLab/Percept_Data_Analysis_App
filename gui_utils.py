import datetime
from datetime import datetime
import pandas as pd
import plotly.io as pio

def translate_param_dict(input_data):  
    def days_difference(date_str, reference_date_str, date_format):
        date = datetime.strptime(date_str, date_format)
        reference_date = datetime.strptime(reference_date_str, date_format)
        return (date - reference_date).days

    # Ensure the date format is consistent throughout the function
    date_format = '%m-%d-%Y'

    # Calculating differences for pre_DBS_example_days
    pre_DBS_example_days = [days_difference(day, input_data['Initial_DBS_programming_date'], date_format) for day in input_data['pre_DBS_example_days']]

    # Calculating differences for post_DBS_example_days
    post_DBS_example_days = [days_difference(day, input_data['Initial_DBS_programming_date'], date_format) for day in input_data['post_DBS_example_days']]

    # Translating hemisphere
    hemisphere = 0
    
    current_day = datetime.now().strftime(date_format)
    if input_data['responder']:
        responder_zone_idx = [
            days_difference(input_data['responder_date'], input_data['Initial_DBS_programming_date'], date_format), 
            days_difference(current_day, input_data['responder_date'], date_format)
        ]
        input_data['responder_zone_idx'] = responder_zone_idx
        input_data['non_responder_idx'] = []  # Ensuring this key is set
    else:
        input_data['responder_zone_idx'] = []  # Ensuring this key is set
        input_data['non_responder_idx'] = [0, days_difference(current_day, input_data['Initial_DBS_programming_date'], date_format)]
        
    # Creating the translated dictionary
    translated_data = {
        "dbs_date": input_data['Initial_DBS_programming_date'],
        "responder_zone_idx": input_data['responder_zone_idx'],
        "non_responder_idx": input_data['non_responder_idx'],
        "subject_name": input_data['subject_name'],
        "pre_DBS_example_days": pre_DBS_example_days,
        "post_DBS_example_days": post_DBS_example_days,
        "hemisphere": hemisphere,
        "cosinor_window_left": 2,
        "cosinor_window_right": 2,
        "include_nonlinear": 0
        
    }

    return translated_data

def add_extension(filename, extension):
    return filename if filename.lower().endswith(extension.lower()) else filename + extension

def save_lin_ar_feature(data, filename):
    ext = filename.split('.')[-1].lower()
    
    if ext == 'json':
        data.to_json(filename, orient='records', lines=True)
    elif ext == 'xlsx':
        data.to_excel(filename, index=False)
    elif ext == 'txt':
        with open(filename, 'w') as f:
            f.write(data.to_string(index=False))
    else:
        # Default to CSV if no extension or an unrecognized extension is provided
        filename = add_extension(filename, '.csv')
        data.to_csv(filename, index=False)

def save_plot(fig, filename):
    ext = filename.split('.')[-1].lower()
    
    if ext in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf']:
        pio.write_image(fig, filename, format=ext)
    else:
        # Default to HTML if no valid extension is provided
        filename = add_extension(filename, '.html')
        fig.write_html(filename, include_plotlyjs='cdn')