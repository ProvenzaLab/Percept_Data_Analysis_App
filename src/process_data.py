import pandas as pd
import numpy as np
import utils.utils as utils
from zoneinfo import ZoneInfo
from datetime import time as dttime
from datetime import datetime, timedelta
import utils.state_utils as state_utils
import json

central_time = ZoneInfo("America/Chicago")


def process_data(
    pt_name: str,
    raw_data: pd.DataFrame,
    patient_dict: dict,
    ark: bool = False,
    max_lag: int = 144,
):

    # Fill outliers: define which filling method(s) you want to use below
    outlier_fill_methods = {
        "naive": utils.fill_outliers,
        "SLOvER+": utils.fill_outliers_threshold,
        "OvER": utils.fill_outliers_overages,
    }

    raw_data.rename(
        columns={"lfp_left": "lfp_left_raw", "lfp_right": "lfp_right_raw"}, inplace=True
    )

    # Put timestamps in datetime format and drop duplicate readings (lots of readings show up in multiple files)
    raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])
    raw_data.drop_duplicates(
        subset=["pt_id", "timestamp", "lead_location"], inplace=True
    )
    raw_data.sort_values(
        ["pt_id", "lead_location", "timestamp"], inplace=True, ignore_index=True
    )

    raw_data["CT_timestamp"] = raw_data["timestamp"].dt.tz_convert(central_time)

    to_seconds = np.vectorize(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
    times = np.array(
        [dttime(hour=h, minute=m) for h in range(24) for m in range(0, 60, 10)]
        + [dttime(0, 0)]
    )
    bins = to_seconds(times[:-1]).astype(int)

    utc_timestamp_secs_since_midnight = to_seconds(
        raw_data["timestamp"].dt.time.values
    ).astype(int)
    utc_dates = raw_data["timestamp"].dt.date
    utc_times = times[
        np.digitize(utc_timestamp_secs_since_midnight, bins, right=False) - 1
    ]
    raw_data["time_bin"] = [
        datetime.combine(utc_dates[i], utc_times[i], tzinfo=ZoneInfo("UTC"))
        for i in range(raw_data.shape[0])
    ]

    # If there are still any duplicates in data streams, eliminate them.
    processed_data = (
        raw_data.groupby(["pt_id", "time_bin", "lead_location"]).tail(1).copy()
    )

    # Add column for days since first DBS activation
    dbs_on_date = state_utils.get_dbs_on_date(patient_dict)
    processed_data["days_since_dbs"] = [
        dt.days for dt in (processed_data["CT_timestamp"].dt.date - dbs_on_date)
    ]

    # Add empty new rows to fill in missing timestamps, and interpolate outliers and missing rows. Label which rows are interpolated.
    processed_data["interpolated"] = False
    added_rows = processed_data.groupby("lead_location", group_keys=False).apply(
        lambda g: utils.fill_holes(g, g.name, dbs_on_date=dbs_on_date),
        include_groups=False,
    )
    if not added_rows.empty:
        processed_data = pd.concat((processed_data, added_rows), ignore_index=True)
    processed_data.sort_values(
        by=["pt_id", "lead_location", "timestamp"], inplace=True, ignore_index=True
    )

    # Fix overvoltages and fill in holes in data using the specified method(s).
    # Adjust outlier filling methods as necessary
    with open(utils.get_data_path("data\\param.json"), "r") as f:
        param_dict = json.load(f)
    name = param_dict["model"]

    func = outlier_fill_methods[name]
    col_dict = {
        "lfp_left_outliers_filled": f"lfp_left_outliers_filled_{name}",
        "lfp_left_filled": f"lfp_left_filled_{name}",
        "lfp_right_outliers_filled": f"lfp_right_outliers_filled_{name}",
        "lfp_right_filled": f"lfp_right_filled_{name}",
    }
    filled_cols = processed_data.groupby(
        ["pt_id", "lead_location"], group_keys=False
    ).apply(
        lambda g: utils.fill_data(
            g, cols_to_fill=["lfp_left_raw", "lfp_right_raw"], outlier_fill_method=func
        ),
        include_groups=False,
    )
    filled_cols.rename(columns=col_dict, errors="ignore", inplace=True)
    processed_data = pd.merge(
        processed_data, filled_cols, how="outer", left_index=True, right_index=True
    )
    filled_cols = [f"lfp_left_filled_{name}"] + [f"lfp_right_filled_{name}"]

    processed_data.dropna(subset=filled_cols, how="all", inplace=True)

    # Mark rows that were filled in with 'interpolated' tag.
    processed_data.loc[
        processed_data[f"lfp_left_filled_{name}"].notna()
        & (
            processed_data["lfp_left_raw"]
            != processed_data[f"lfp_left_outliers_filled_{name}"]
        ),
        "interpolated",
    ] = True
    processed_data.loc[
        processed_data[f"lfp_right_filled_{name}"].notna()
        & (
            processed_data["lfp_right_raw"]
            != processed_data[f"lfp_right_outliers_filled_{name}"]
        ),
        "interpolated",
    ] = True

    # Z score LFP data within each day.
    processed_data = processed_data.reset_index(drop=True)
    groups = processed_data.groupby(
        ["pt_id", pd.Grouper(key="CT_timestamp", freq="D"), "lead_location"],
        group_keys=False,
    )
    zscored_data = groups.apply(
        lambda g: utils.zscore_group(g, cols_to_zscore=filled_cols),
        include_groups=False,
    )
    zscored_cols = zscored_data.columns
    processed_data = pd.merge(
        processed_data, zscored_data, how="outer", left_index=True, right_index=True
    )

    # Find and label intervals of consecutive readings every 10 minutes. This is needed to correctly formulate lags.
    groups = processed_data.groupby("lead_location", group_keys=False)
    for col in zscored_cols:
        contig_colname = "contig_left" if "left" in col else "contig_right"
        if contig_colname in processed_data.columns:
            continue
        processed_data = processed_data.join(
            groups.apply(
                lambda g: utils.get_contig(
                    g, col_to_check=col, contig_colname=contig_colname
                ),
                include_groups=False,
            ),
            how="left",
        )

    # Add new lag1 column for autoregression.
    groups_left = processed_data.groupby(
        ["lead_location", "contig_left"], group_keys=False
    )
    groups_right = processed_data.groupby(
        ["lead_location", "contig_right"], group_keys=False
    )
    for i, name in enumerate(zscored_cols):
        if i == 0:
            join_df = groups_left.apply(
                lambda g: pd.DataFrame({f"{name}_lag_1": g[name].shift(periods=1)}),
                include_groups=False,
            )
            if join_df.empty:
                processed_data[f"{name}_lag_1"] = np.nan
            else:
                processed_data = processed_data.join(join_df, how="left")

            if ark:
                for k in range(2, max_lag + 1):
                    join_df = groups_left.apply(
                        lambda g: pd.DataFrame(
                            {f"{name}_lag_{k}": g[name].shift(periods=k)}
                        ),
                        include_groups=False,
                    )
                    if join_df.empty:
                        processed_data[f"{name}_lag_{k}"] = np.nan
                    else:
                        processed_data = processed_data.join(join_df, how="left")

        else:
            join_df = groups_right.apply(
                lambda g: pd.DataFrame({f"{name}_lag_1": g[name].shift(periods=1)}),
                include_groups=False,
            )
            if join_df.empty:
                processed_data[f"{name}_lag_1"] = np.nan
            else:
                processed_data = processed_data.join(join_df, how="left")

            if ark:
                for k in range(2, max_lag + 1):
                    join_df = groups_right.apply(
                        lambda g: pd.DataFrame(
                            {f"{name}_lag_{k}": g[name].shift(periods=k)}
                        ),
                        include_groups=False,
                    )
                    if join_df.empty:
                        processed_data[f"{name}_lag_{k}"] = np.nan
                    else:
                        processed_data = processed_data.join(join_df, how="left")

    state_labels = state_utils.get_state_labels(processed_data, patient_dict)
    processed_data = processed_data.drop(
        columns=state_labels.columns, errors="ignore"
    ).join(state_labels)

    # Add column for patient's ID
    processed_data["pt_id"] = pt_name

    # Drop any remaining duplicate readings and sort processed_data.
    processed_data.drop_duplicates(
        [
            "pt_id",
            "lfp_left_raw",
            "lfp_right_raw",
            "stim_left",
            "stim_right",
            "lead_location",
            "time_bin",
        ],
        inplace=True,
        ignore_index=True,
    )
    processed_data.sort_values(
        by=["pt_id", "lead_location", "timestamp"], inplace=True, ignore_index=True
    )

    # Mark outliers.
    processed_data["is_outlier_left"] = (
        processed_data["lfp_left_raw"] >= ((2**32) - 1) / 60
    ) & (processed_data["lfp_left_raw"].notna())
    processed_data["is_outlier_right"] = (
        processed_data["lfp_right_raw"] >= ((2**32) - 1) / 60
    ) & (processed_data["lfp_right_raw"].notna())
    processed_data["time_bin_time"] = processed_data["time_bin"].dt.time

    # Print outlier composition
    vcvs_df = processed_data.query('lead_location == "VC/VS"')
    left_outliers = (
        vcvs_df["is_outlier_left"].sum() / vcvs_df["is_outlier_left"].count() * 100
    )
    right_outliers = (
        vcvs_df["is_outlier_right"].sum() / vcvs_df["is_outlier_right"].count() * 100
    )
    # print(f'{pt_name} Left Outlier %: {left_outliers}')
    # print(f'{pt_name} Right Outlier %: {right_outliers}')

    return processed_data
