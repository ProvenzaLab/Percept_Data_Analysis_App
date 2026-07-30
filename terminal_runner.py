from src.pipeline import run_pipeline
import utils.gui_utils as utils
import pandas as pd
import numpy as np
from datetime import time as dttime
import os

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
    """Run the terminal version of the percept_data analysis app.

    Should be used as a toy exploration of the pipeline. Edit
    ``data/patient_info.json`` to change which patients are processed.
    """
    import multiprocessing
    import json

    with open("data/patient_info.json", "r") as f:
        patient_dict = json.load(f)

    with open("data/param.json", "r") as f:
        param_dict = json.load(f)

    queue = multiprocessing.Queue()
    run_pipeline(patient_dict, queue)
    result = queue.get()

    if result is None:
        print("Pipeline failed; check logs above for the underlying error.")
        return

    df_final, pt_changes_df = result

    if df_final.empty:
        print("No data was produced for any patient.")
        return

    print(f"{len(df_final['pt_id'].unique())} patient(s) processed.")

    # Export data into excel files
    if export:
        file_path = utils.open_save_dialog(os.curdir(), "Save Data", "")
        if file_path:
            export_raw_data(df_final, file_path)
            export_res_stats(df_final, file_path)


if __name__ == "__main__":
    main(export=False)