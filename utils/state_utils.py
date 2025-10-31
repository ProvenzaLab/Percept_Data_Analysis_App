import pandas as pd
from datetime import datetime

def get_dbs_on_date(pt_data):
    """
    Retrieves date that stimulation was turned on from info dict.

    Parameters:
    - pt_data (dictionary): patient info dictionary containing DBS on date in string format.

    Returns:
    - dbs_on_date (datetime.date): DBS on date in the datetime.date type.
    """

    return datetime.strptime(pt_data["dbs_date"], "%Y-%m-%d").date()


def get_state_labels(pt_df: pd.DataFrame, patient_dict: dict):
    """
    Calculate the patient's state at each time point based on their scale scores.
    State key:
        0 = Pre-DBS
        1 = Disinhibited
        2 = Non-responder
        3 = Responder
        4 = Unknown

    Parameters:
    - pt_df (pd.DataFrame): DataFrame containing all data for this patient.
    - patient_dict: Dictionary contatining response dates and status about the patient
    Returns:
    - pt_df (pd.DataFrame): Updated DataFrame containing state labels and strings.
    """

    # Assign all data points a default starting label of 5 ('unknown').
    pt_df["state_label"] = 4  # unknown/unlabeled
    pt_df["state_label_str"] = "Unknown"

    # Assign pre-DBS points a label of 0.
    pt_df.loc[pt_df["days_since_dbs"] < 0, "state_label"] = 0  # pre-DBS
    pt_df.loc[pt_df["days_since_dbs"] < 0, "state_label_str"] = "Pre-DBS"  # pre-DBS

    if patient_dict["response_status"]:
        if type(patient_dict["response_date"]) != int:
            response_date = (
                datetime.strptime(patient_dict["response_date"], "%Y-%m-%d").date()
                - get_dbs_on_date(patient_dict)
            ).days
        else:
            response_date = patient_dict["response_date"]
        pt_df.loc[pt_df["days_since_dbs"] >= response_date, "state_label"] = (
            3  # Responder
        )
        pt_df.loc[pt_df["days_since_dbs"] >= response_date, "state_label_str"] = (
            "Responder"
        )
    else:
        pt_df.loc[pt_df["days_since_dbs"] >= 0, "state_label"] = 2  # Non-responder
        pt_df.loc[pt_df["days_since_dbs"] >= 0, "state_label_str"] = "Non-responder"

    try:
        disinhibited_dates = patient_dict["disinhibited_dates"]
        for i, date in enumerate(disinhibited_dates):
            if type(date) != int:
                date = datetime.strptime(date, "%Y-%m-%d").date()
                date = (date - get_dbs_on_date(patient_dict)).days
                disinhibited_dates[i] = date

        pt_df.loc[
            (
                (pt_df["days_since_dbs"] >= disinhibited_dates[0])
                & (pt_df["days_since_dbs"] <= disinhibited_dates[1])
            ),
            "state_label",
        ] = 1  # Disinhibited
        pt_df.loc[
            (
                (pt_df["days_since_dbs"] >= disinhibited_dates[0])
                & (pt_df["days_since_dbs"] <= disinhibited_dates[1])
            ),
            "state_label_str",
        ] = "Disinhibited"
    except Exception:
        pass

    return pt_df
