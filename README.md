# Percept Data Analysis App

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

Example values for these fields are provided in param.json

