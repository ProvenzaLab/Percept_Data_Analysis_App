"""Pipeline orchestration used by both the GUI worker and the CLI runner.

``run_pipeline`` reads ``data/param.json`` once and fans the generate →
process → model chain out across every patient in ``patient_dict`` using a
process pool, since each patient is processed independently. Progress and
results are pushed onto ``result_queue`` as typed dict messages so a caller
(e.g. the GUI) can report per-patient status while the batch is still
running:

- ``{"type": "started", "total": N}`` once, before any patient starts.
- ``{"type": "progress", "pt_id": pt, "success": bool, "error": str | None}``
  as each patient finishes, in whatever order the pool completes them.
- ``{"type": "done", "df_final": df, "pt_changes_df": df, "failed": [(pt_id, error), ...]}``
  once, after every patient has finished.
- ``{"type": "error", "message": str}`` if the run itself fails outside of
  any single patient's processing.
"""

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

import src.generate_raw as generate_raw
import src.process_data as process_data
import src.model_data as model_data
from utils.utils import get_data_path


def _process_one_patient(pt: str, patient_info: dict, param_dict: dict):
    """Run generate → process → model for a single patient.

    Returns a ``(pt, success, df_w_preds, param_changes, error)`` tuple.
    ``df_w_preds``/``param_changes`` are ``None`` and ``error`` is a message
    string on failure. Runs in a worker process, so this must stay a
    module-level function (picklable) and must not raise — every failure
    mode is caught and reported back through the tuple instead, so one bad
    patient can't take down the rest of the batch.
    """
    try:
        raw_df, param_changes = generate_raw.generate_raw(pt, patient_info)
    except (TypeError, ValueError) as e:
        return pt, False, None, None, str(e)

    try:
        processed_data = process_data.process_data(
            pt,
            raw_df,
            patient_info,
            ark=param_dict["ark"],
            max_lag=param_dict["lags"] if param_dict["ark"] else 1,
        )
        df_w_preds = model_data.model_data(
            processed_data,
            use_constant=bool(param_dict["ark"]),
            ark=param_dict["ark"],
            max_lag=param_dict["lags"] if param_dict["ark"] else 1,
        )
    except Exception as e:
        return pt, False, None, None, f"Processing failed: {e}"

    return pt, True, df_w_preds, param_changes, None


def run_pipeline(patient_dict: dict, result_queue) -> None:
    """Execute the analysis pipeline across every patient in ``patient_dict``.

    Parameters
    ----------
    patient_dict : dict
        Mapping of patient_id → patient_info (as stored in
        ``data/patient_info.json``) for the patients to process. Callers
        that cache previously-processed patients should pass only the
        patients that still need processing.
    result_queue : multiprocessing.Queue
        Queue onto which typed status messages are put; see the module
        docstring for the message protocol.
    """
    try:
        with open(get_data_path("data\\param.json"), "r") as f:
            param_dict = json.load(f)

        total = len(patient_dict)
        result_queue.put({"type": "started", "total": total})

        if total == 0:
            result_queue.put(
                {
                    "type": "done",
                    "df_final": pd.DataFrame(),
                    "pt_changes_df": pd.DataFrame(),
                    "failed": [],
                }
            )
            return

        df_parts = []
        changes_parts = []
        failed = []

        workers = min(total, os.cpu_count() or 4)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(_process_one_patient, pt, info, param_dict): pt
                for pt, info in patient_dict.items()
            }
            for future in as_completed(futures):
                pt = futures[future]
                try:
                    pt_id, success, df_w_preds, param_changes, error = future.result()
                except Exception as e:
                    pt_id, success, df_w_preds, param_changes, error = (
                        pt,
                        False,
                        None,
                        None,
                        str(e),
                    )

                if success:
                    df_parts.append(df_w_preds)
                    changes_parts.append(param_changes)
                else:
                    failed.append((pt_id, error))

                result_queue.put(
                    {
                        "type": "progress",
                        "pt_id": pt_id,
                        "success": success,
                        "error": error,
                    }
                )

        df_final = pd.concat(df_parts, ignore_index=True) if df_parts else pd.DataFrame()
        pt_changes_df = (
            pd.concat(changes_parts, ignore_index=True) if changes_parts else pd.DataFrame()
        )

        result_queue.put(
            {
                "type": "done",
                "df_final": df_final,
                "pt_changes_df": pt_changes_df,
                "failed": failed,
            }
        )

    except Exception as e:
        result_queue.put({"type": "error", "message": str(e)})
