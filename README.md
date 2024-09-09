# Percept Data Analysis App

## Overview

The Percept Desktop App is designed to provide an intuitive user interface for running the percept-data analysis pipeline for OCD patients, as detailed in this [paper](https://www.nature.com/articles/s41591-024-03125-0). The original program was developed using a combination of MATLAB and Python, as seen in the [PerceptDataAnalysis repository](https://github.com/shethlab/PerceptDataAnalysis). This application translates all the code into Python and uses a Python-based library to create the GUI. The source code and instructions for downloading the app can be found on this GitHub repo

## User Manual

### Key Fields

- `Initial_DBS_programming_date`: The date the DBS treatment started, entered in the format MM-DD-YYYY.
- `Subject_name`: The codename used to track a specific patient (e.g., '009').
- `Pre_DBS_example_days`: Enter two dates before DBS treatment to see the interval plotted, in the format MM-DD-YYYY, MM-DD-YYYY.
- `Post_DBS_example_days`: Enter two dates after DBS treatment to see the interval plotted, in the format MM-DD-YYYY, MM-DD-YYYY.
- `Responder`: Indicates whether the subject has achieved clinical response as noted by YBOCS criteria (e.g., 'yes' or 'no').
- `Responder_date`: If 'yes' was selected for Responder, provide the date when the patient reached clinical response in the format MM-DD-YYYY.

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
- If you enter an unsupported extension, such as `plot_right_008.bmp`, the app will still save it as `plot_right_008.html` by default

#### Export Linear AR Feature

- Linear AR features can be exported in `csv`, `xlsx`, `json`, `tsv`, or `txt` formats.
- Similar to plot downloads, when you enter a filename in the file dialog without an extension, it will default to `.csv`.
- If you specify an extension, the file will be saved in the corresponding format, unknown file formats will default `.csv`.

### Installation and Demo Video:
- [Video](https://drive.google.com/file/d/1tWAAfF2GR7SGf6W4wstNonslh4T7LCWn/view)


## Developer Guide

This section is intended for developers looking to modify or extend the functionality of the Percept Data Analysis App. The app is divided into two primary components:

1. **Core Analysis Pipeline**
2. **GUI Interface**

### Core Analysis Pipeline

The Core Analysis Pipeline consists of the following key files:

- `generate_data.py`: Generates a data_struct from Medtronic Percept data files, which is used in subsequent analyses.
- `calc_circadian.py`: Processes the data_struct to append basic statistical results (e.g., cosinor analysis, sample entropy).
- `calc_circadian_advanced.py`: Processes the data_struct to append advanced statistical results (e.g., linear AR, nonlinear AR).
- `plotting_utils.py`: Generates a summary plot encapsulating the most critical information from the processed data_struct.
- `utils.py`: Provides various helper methods utilized by the above files.

#### User Defined Hyper-Parameters

To run the Core Analysis Pipeline, users must specify certain hyperparameters, divided into `Required` and `Optional` categories.

**Required Parameters:**

- `dbs_date`: The start date of DBS treatment, in the format MM-DD-YYYY.
- `subject_name`: A codename for tracking a specific patient (e.g., '009').
- `responder_zone_idx`: A tuple containing two values: (responder_date - `dbs_date`, current_date - responder_date). Use an empty tuple if the patient is a non-responder.
- `non_responder_idx`: A tuple containing two values: (0, current_date - `dbs_date`). Use an empty tuple if the patient is a responder.
- `pre_DBS_example_days`: A tuple with two integers representing an interval (relative to `dbs_date`) to be displayed in the pre-DBS zoomed subplot generated by `plot_metrics.py`. For example, if `pre_dbs_example_days` is [-12, -9] and `dbs_date` is 12-22-2023, the zoomed interval is [12-10-2023, 12-13-2023].
- `post_DBS_example_days`: A tuple with two integers representing an interval (relative to `dbs_date`) to be displayed in the post-DBS zoomed subplot generated by `plot_metrics.py`. For example, if `post_dbs_example_days` is [85, 87] and `dbs_date` is 12-22-2023, the zoomed interval is [03-16-2024, 03-18-2024].
- `hemisphere`: An integer representing the hemisphere being plotted in `plot_metrics.py` (0 for the left hemisphere, 1 for the right hemisphere).

**Optional Parameters:**

- `cosinor_window_left`: An integer representing the number of days before the day of interest to include in the cosinor calculation window. Default value is `2`.
- `cosinor_window_right`: An integer representing the number of days after the day of interest to include in the cosinor calculation window. Default value is `2`.
- `include_nonlinear`: A boolean flag indicating whether to run the non-linear analysis (neural net). Default value is `False`.

Example values for these parameters are provided in `param.json`.

#### Running the Core Analysis Pipeline

Typically, the Core Analysis Pipeline is executed in the following order:

1. `generate_data.py`
2. `calc_circadian.py`
3. `plotting_utils.py`

The `calc_circadian_advanced.py` script is implicitly called within `calc_circadian.py`.

For a simple example of running this execution pipeline, refer to `terminal_runner.py`. This file provides a basic script to run the data analysis and display the plots generated by `plotting_utils.py`. To modify the hyperparameters, update them in `param.json`.

For specific implementation details, refer to the documentation and comments within these scripts.

### GUI Interface

The GUI Interface is primarily built using two files: `app.py` and `gui_utils.py`.

- `app.py`: This is the main file responsible for generating the GUI, including widgets, textboxes, and other UI elements.
- `gui_utils.py`: A utility file used by `app.py` to perform tasks such as data export, validation, and transformations.

Documentation for the GUI component is minimal, as it is designed to serve as a flexible abstraction layer for the Core Analysis Pipeline. Developers are encouraged to customize the GUI to fit specific needs. The GUI can be replaced or modified, as long as it can interface with the Core Analysis Pipeline and correctly format the user-defined hyperparameters.


