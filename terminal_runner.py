from src.generate_raw import generate_raw
from src.process_data import process_data
from src.model_data import model_data
from utils.plotting_utils import plot_metrics
import pandas as pd
import numpy as np
import json
import utils.gui_utils as utils
import os
from datetime import time as dttime

# Export R² and Residual stats
def export_res_stats(df_final, filename):
    pt_summary_stats = {}

    for pt in df_final["pt_id"].unique():
        # Subset patient data
        pt_data = df_final.query(
            'pt_id == @pt and lead_location == "VC/VS"'
        ).drop_duplicates(subset=["days_since_dbs"])

        # Initialize patient summary stats dataframe
        pt_df = pd.DataFrame()

        # Compute and assign summary stats for each day
        pt_df["days_since_dbs"] = pt_data["days_since_dbs"]
        pt_df["State_Label"] = pt_data["state_label"]
        pt_df["r2"] = pt_data["lfp_left_day_r2_OvER"]
        pt_df["res_var"] = pt_data["lfp_left_residual_var_OvER"]
        pt_df["lambda_2.5"] = pt_data["lfp_left_lambda_25_OvER"]
        pt_df["mu_0"] = pt_data["lfp_left_mu_0_OvER"]
        pt_df["mu_2.5"] = pt_data["lfp_left_mu_25_OvER"]
        pt_df["sigma_2.5"] = pt_data["lfp_left_sigma_25_OvER"]
        pt_df["raw_var"] = [
            np.nanvar(
                df_final.query(
                    'pt_id == @pt and lead_location == "VC/VS" and days_since_dbs == @day'
                )["lfp_left_outliers_filled_OvER"]
            )
            for day in pt_df.days_since_dbs
        ]

        # Add summary stats to dictionary
        pt_summary_stats[pt] = pt_df

    with pd.ExcelWriter(
        filename, mode="a", if_sheet_exists="replace", engine="openpyxl"
    ) as writer:
        for pt in df_final["pt_id"].unique():
            pt_summary_stats[pt].to_excel(writer, sheet_name=f"{pt}")

    return

# Export Raw LFP, Predicted LFP, and Residual LFP
def export_raw_data(df_final, filename):
    times = [dttime(i // 60, i % 60) for i in range(0, 1440, 10)]

    df_vcvs = df_final[
        (df_final["lead_location"] == "VC/VS") | (df_final["lead_location"] == "OTHER")
    ]
    pt_lfp_dfs = {}
    pt_pred_dfs = {}
    pt_res_dfs = {}

    for pt in df_final["pt_id"].unique():
        pt_df = df_vcvs[df_vcvs["pt_id"] == pt]
        days = pt_df["days_since_dbs"].drop_duplicates().dropna()

        pt_lfp = pd.DataFrame(columns=days, index=times)
        pt_pred = pd.DataFrame(columns=days, index=times)
        pt_res = pd.DataFrame(columns=days, index=times)

        for day in days:
            day_df = pt_df[pt_df["days_since_dbs"] == day]
            day_df.loc[:, "time_bin"] = day_df["time_bin"] - pd.Timedelta(6, unit="h")
            day_df["time_bin"] = day_df["time_bin"].dt.time
            for time_bin, value in day_df[
                ["time_bin", "lfp_left_z_scored_OvER"]
            ].values:
                if time_bin in pt_lfp.index:
                    pt_lfp.loc[time_bin, day] = value
            for time_bin, value in day_df[["time_bin", "lfp_left_preds_OvER"]].values:
                if time_bin in pt_pred.index:
                    pt_pred.loc[time_bin, day] = value
            for time_bin, value in day_df[
                ["time_bin", "lfp_left_residuals_OvER"]
            ].values:
                if time_bin in pt_res.index:
                    pt_res.loc[time_bin, day] = value

        pt_lfp_dfs[pt] = pt_lfp
        pt_pred_dfs[pt] = pt_pred
        pt_res_dfs[pt] = pt_res

    with pd.ExcelWriter(
        filename, mode="a", if_sheet_exists="replace", engine="openpyxl"
    ) as writer:
        for pt in df_final["pt_id"].unique():
            pt_lfp_dfs[pt].to_excel(writer, sheet_name=f"{pt}_LFP")
            pt_pred_dfs[pt].to_excel(writer, sheet_name=f"{pt}_Pred")
            pt_res_dfs[pt].to_excel(writer, sheet_name=f"{pt}_Res")

    return


def main(export):
    """_summary_
    Used to run the terminal version of the percept_data analysis app.
    Should be used as a toy exploration of the pipeline, change parameters in patient_info.json to see results.
    """

    with open("data/patient_info.json", "r") as f:
        patient_dict = json.load(f)

    with open("data/param.json", "r") as f:
        param_dict = json.load(f)

    pt = patient_dict[0]  # Process data from first patient in patient_info.json
    try:
        raw_df, param_changes = generate_raw.generate_raw(pt, patient_dict[pt])

    except TypeError or ValueError as e:
        print(f"Unable to retrieve data for pateint {pt}")
        return

    processed_data = process_data.process_data(
        pt,
        raw_df,
        patient_dict[pt],
        ark=param_dict["ark"],
        max_lag=param_dict["lags"] if param_dict["ark"] else 1,
    )

    df_w_preds = model_data.model_data(
        processed_data,
        use_constant=False if not param_dict["ark"] else True,
        ark=param_dict["ark"],
        max_lag=param_dict["lags"] if param_dict["ark"] else 1,
    )

    pt_changes_df = pd.concat([pt_changes_df, param_changes], ignore_index=True)

    df_final = pd.concat([df_final, df_w_preds], ignore_index=True)

    print(f"{pt} done")

    # Plot the metrics
    fig = plot_metrics(
        df_w_preds,
        pt,
        'left',
        pt_changes_df,
        show_changes=False,
        patients_dict=patient_dict,
        param_dict=param_dict,
    )

    fig.show()

    # Export data into excel files
    if export:
        file_path = utils.open_save_dialog(os.curdir(), "Save Data", "")
        if file_path:
            export_raw_data(df_final, file_path)
            export_res_stats(df_final, file_path)


if __name__ == "__main__":
    main(export=False)
