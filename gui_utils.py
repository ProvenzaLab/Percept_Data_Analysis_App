import datetime
from datetime import datetime


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
        "hemisphere": hemisphere
    }

    return translated_data

def add_extension(filename, extension):
    return filename if filename.lower().endswith(extension.lower()) else filename + extension