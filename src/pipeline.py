"""Pipeline orchestration used by both the GUI worker and the CLI runner.

``run_pipeline`` reads ``data/param.json`` once and runs the full
generate → process → model chain for every patient in
``patient_dict``. Results are pushed onto ``result_queue`` as a tuple
``(df_final, pt_changes_df)``; on error, ``None`` is pushed so callers
can distinguish a hard failure from an empty-but-valid output.
"""

import json

import pandas as pd

import src.generate_raw as generate_raw
import src.process_data as process_data
import src.model_data as model_data
from utils.utils import get_data_path


def run_pipeline(patient_dict: dict, result_queue) -> None:
    """Execute the full analysis pipeline across all patients.

    Parameters
    ----------
    patient_dict : dict
        Mapping of patient_id → patient_info (as stored in
        ``data/patient_info.json``).
    result_queue : multiprocessing.Queue
        Queue onto which the result ``(df_final, pt_changes_df)`` is
        put on success, or ``None`` on hard failure.
    """
    try:
        df_final = pd.DataFrame()
        pt_changes_df = pd.DataFrame()

        with open(get_data_path("data\\param.json"), "r") as f:
            param_dict = json.load(f)

        for pt in patient_dict.keys():
            try:
                raw_df, param_changes = generate_raw.generate_raw(
                    pt, patient_dict[pt]
                )
            except (TypeError, ValueError) as e:
                print(f"Unable to retrieve data for patient {pt}: {e}")
                continue

            processed_data = process_data.process_data(
                pt,
                raw_df,
                patient_dict[pt],
                ark=param_dict["ark"],
                max_lag=param_dict["lags"] if param_dict["ark"] else 1,
            )

            df_w_preds = model_data.model_data(
                processed_data,
                use_constant=bool(param_dict["ark"]),
                ark=param_dict["ark"],
                max_lag=param_dict["lags"] if param_dict["ark"] else 1,
            )

            pt_changes_df = pd.concat(
                [pt_changes_df, param_changes], ignore_index=True
            )

            df_final = pd.concat([df_final, df_w_preds], ignore_index=True)

        result_queue.put((df_final, pt_changes_df))

    except Exception as e:
        print(f"Error in run_pipeline: {e}")
        result_queue.put(None)
