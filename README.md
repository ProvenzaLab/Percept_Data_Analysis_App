# Percept Data Analysis App

## Overview

The Percept Desktop App is designed to provide an intuitive user interface for running the percept-data analysis pipeline for OCD patients, as detailed in this [paper](https://www.nature.com/articles/s41591-024-03125-0). The original program was developed using a combination of MATLAB and Python, as seen in the [PerceptDataAnalysis repository](https://github.com/shethlab/PerceptDataAnalysis). This application translates all the code into Python and uses a Python-based library to create the GUI. The source code and instructions for downloading the app can be found on GitHub at this URL: https://tobedetermined.

## User Manual

### Key Fields

- **Initial_DBS_programming_date**: The date the DBS treatment started, entered in the format MM-DD-YYYY.
- **Subject_name**: The codename used to track a specific patient (e.g., '009').
- **pre_DBS_example_days**: Enter two dates before DBS treatment to see the interval plotted, in the format MM-DD-YYYY, MM-DD-YYYY.
- **post_DBS_example_days**: Enter two dates after DBS treatment to see the interval plotted, in the format MM-DD-YYYY, MM-DD-YYYY.
- **Responder**: Indicates whether the subject has achieved clinical response as noted by YBOCS criteria (e.g., 'yes' or 'no').
- **Responder_date**: If 'yes' was selected for Responder, provide the date when the patient reached clinical response in the format MM-DD-YYYY.

### Features
- **Plot Metrics**: Displays various physiological and linear AR model metrics. More information can be found in the original paper linked above.
- **Download Plot**: The app can download plots as a variety of different files formats using the "Download Plot" button.
- **Export Data**: The raw linear-AR R2 values can be exported variety of different files formats using the “export linAR button”.

### Data Export Guide

Data exporting is a key feature of the Percept Data Analysis App and was designed to make sharing and analyzing your results as seamless as possible.

#### Download Plot

- You can download plots in various formats, including `html`, `png`, `jpg`, `jpeg`, `webp`, `svg`, and `pdf`.
- When saving a plot, a file dialog will pop up, allowing you to choose a target directory and enter a filename. If you don't specify a file extension, the plot will be saved as `html` by default, enabling interactive viewing in your browser.
- For example, if you name your file `plot_right_008`, it will automatically save as `plot_right_008.html`. To save it in a different format, simply add the desired extension, like `plot_right_008.jpg`.
- If you enter an unsupported extension, such as `plot_right_008.bmp`, the app will still save it as `plot_right_008.html` to ensure compatibility.

#### Export Linear AR Feature

- Linear AR features can be exported in `csv`, `xlsx`, `json`, or `txt` formats.
- Similar to plot downloads, when you enter a filename in the file dialog without an extension, it will default to `.csv`.
- If you specify an extension, the file will be saved in the corresponding format, unknown file formats will default `.csv`.

### Demo video:
- to be linked


## Internal Development

This section provides an overview of the internal development process for the Percept Data Analysis App.

### Required Files

For internal development, the primary code files you need to interface with are:

- `generate_data.py`
- `calc_circadian.py`
- `calc_circadian_advanced.py`
- `plotting_utils.py`

You can learn how to call these files using the simplified `terminal_runner.py`. All user-needed parameters to run the Percept Analysis pipeline can be found in `param.json`.

### Required Parameters

The parameters that must be specified by the user in the `param.json` file are:

- `dbs_date`
- `responder_zone_idx`
- `non_responder_idx`
- `subject_name`
- `pre_DBS_example_days`
- `post_DBS_example_days`
- `hemisphere`

Example values for these fields are provided in `param.json`.
