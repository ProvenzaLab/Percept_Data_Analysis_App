import datetime
from datetime import datetime
import pandas as pd
import plotly.io as pio
import tempfile
from PySide6.QtWidgets import QFileDialog

DATE_FORMAT = '%m-%d-%Y'

def translate_param_dict(input_data):  
    def days_difference(date_str, reference_date_str):
        date = datetime.strptime(date_str, DATE_FORMAT)
        reference_date = datetime.strptime(reference_date_str, DATE_FORMAT)
        return (date - reference_date).days

    pre_DBS_example_days = [days_difference(day, input_data['Initial_DBS_programming_date']) for day in input_data['pre_DBS_example_days']]
    post_DBS_example_days = [days_difference(day, input_data['Initial_DBS_programming_date']) for day in input_data['post_DBS_example_days']]

    hemisphere = 0
    
    current_day = datetime.now().strftime(DATE_FORMAT)
    if input_data['responder']:
        responder_zone_idx = (
            days_difference(input_data['responder_date'], input_data['Initial_DBS_programming_date']), 
            days_difference(current_day, input_data['responder_date'])
        )
        input_data['responder_zone_idx'] = responder_zone_idx
        input_data['non_responder_idx'] = []
    else:
        input_data['responder_zone_idx'] = []
        input_data['non_responder_idx'] = (0, days_difference(current_day, input_data['Initial_DBS_programming_date']))
        
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
        "include_nonlinear": 0,
        "responder_date": input_data['responder_date']
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
    elif ext == 'tsv':
        data.to_csv(filename, sep='\t', index=False)
    elif ext == 'txt':
        with open(filename, 'w') as f:
            f.write(data.to_string(index=False))
    else:
        filename = add_extension(filename, '.csv')
        data.to_csv(filename, index=False)

def save_plot(fig, filename):
    ext = filename.split('.')[-1].lower()
    
    if ext in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf']:
        pio.write_image(fig, filename, format=ext)
    else:
        filename = add_extension(filename, '.html')
        fig.write_html(filename, include_plotlyjs='cdn')

def open_file_dialog(parent):
    file_dialog = QFileDialog()
    return file_dialog.getOpenFileNames(parent, 'Select patient JSON files', '', 'JSON files (*.json)')[0]

def open_save_dialog(parent, title, default_filter):
    file_path, _ = QFileDialog.getSaveFileName(parent, title, "", default_filter)
    return file_path

def create_temp_plot(fig):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
        temp_file.write(fig.to_html(include_plotlyjs='cdn').encode('utf-8'))
        return temp_file.name

def prepare_export_data(percept_data, param_dict):
    linAR_r2 = percept_data['linearAR_R2'][param_dict['subject_name']][param_dict['hemisphere']]
    days = percept_data['days'][param_dict['subject_name']][param_dict['hemisphere']]
    export_dict = {
        'Days_since_DBS': days.tolist(),
        'R2_values': linAR_r2.tolist()
    }
    lin_ar_df = pd.DataFrame(export_dict)
    lin_ar_df['activation_state'] = lin_ar_df['Days_since_DBS'].apply(
        lambda x: (
            'Responder' if len(param_dict['responder_zone_idx']) > 0 and x > param_dict['responder_zone_idx'][0]
            else 'Pre-DBS' if x < 0
            else 'Chronic State'
        )   
    )
    return lin_ar_df

def validate_date(date_str):
    try:
        datetime.strptime(date_str, DATE_FORMAT)
        return True
    except ValueError:
        return False
