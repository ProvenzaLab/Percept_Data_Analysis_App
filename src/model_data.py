import pandas as pd
from utils.utils import get_data_path
import utils.model_utils as model_utils
import json


def model_data(
    df: pd.DataFrame,
    window_size=3,
    use_constant=False,
    ark=False,
    max_lag=144,
):
    with open(get_data_path("data\\param.json"), "r") as f:
        param_dict = json.load(f)
    model = param_dict["model"]
    window_size = int(param_dict["Window size"])
    hemis = ["left", "right"]

    df["constant"] = 1.0
    df_w_preds = df.copy()

    for hemi in hemis:
        lag_prefix = f"lfp_{hemi}_z_scored_{model}_lag_"
        all_lags = [f"{lag_prefix}{i}" for i in range(1, max_lag + 1)]
        ar_features = (
            (
                [f"lfp_{hemi}_z_scored_{model}_lag_1"]
                + (["constant"] if use_constant else [])
            )
            if not ark
            else all_lags.copy()
        )
        target = f"lfp_{hemi}_z_scored_{model}"

        # Filter for patient/lead group
        pt_groups = df_w_preds.groupby(
            ["pt_id", "lead_location"], group_keys=False
        )

        hemi_results_df = pt_groups.apply(
            lambda g: model_utils.apply_sliding_window(
                g, ar_features, target, window_size=window_size
            ),
            include_groups=False,
        )

        df_w_preds = pd.merge(
            df_w_preds, hemi_results_df, how="outer", left_index=True, right_index=True
        )

    return df_w_preds
