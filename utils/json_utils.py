import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def get_json_filenames(root_dir):
    """
    Retrieves a list of all json files containing LFP data in the provided root directory.

    Parameters:
        root_dir (pathlib.Path): Directory from which to get JSON descendents.

    Returns:
        all_files (list): JSON descendents of root directory containing LFP data.
    """

    root_dir = Path(root_dir)
    if not root_dir.exists():
        return []

    all_files = []
    if np.any([f.is_dir() for f in root_dir.iterdir()]):
        for subdir in root_dir.iterdir():
            if not subdir.is_dir():
                continue
            p = subdir.glob("[!.]*.json")
            # all_files.extend([x for x in p if x.name[:3] == x.parents[2].name])
            all_files.extend(p)
    else:
        p = root_dir.glob("[!.]*.json")
        all_files.extend(p)
    return all_files


def chronic_lfp_from_json(js, filename) -> pd.DataFrame:
    """
    Processes a Percept JSON file to extract chronic LFP power data recorded with BrainSense Timeline and returns it as a DataFrame.

    Parameters:
        filename (pathlib.Path): Path to the JSON file.

    Returns:
        processed_df (pd.DataFrame): DataFrame containing the processed LFP data.
    """

    if "DiagnosticData" not in js or "LFPTrendLogs" not in js["DiagnosticData"]:
        return pd.DataFrame()

    left_lead_config = next(
        (
            d
            for d in js["LeadConfiguration"]["Final"]
            if d["Hemisphere"] == "HemisphereLocationDef.Left"
        ),
        None,
    )
    right_lead_config = next(
        (
            d
            for d in js["LeadConfiguration"]["Final"]
            if d["Hemisphere"] == "HemisphereLocationDef.Right"
        ),
        None,
    )
    left_lead_location = (
        left_lead_config["LeadLocation"].split(".")[-1].upper()
        if left_lead_config
        else None
    )
    right_lead_location = (
        right_lead_config["LeadLocation"].split(".")[-1].upper()
        if right_lead_config
        else None
    )
    left_lead_model = (
        left_lead_config["Model"].split(".")[-1].upper() if left_lead_config else None
    )
    right_lead_model = (
        right_lead_config["Model"].split(".")[-1].upper() if right_lead_config else None
    )

    if left_lead_location in ["VCVS", "VC/VS", "VC", "VS", "AIC", "ALIC", "BNST"]:
        left_lead_location = "VC/VS"
    if right_lead_location in ["VCVS", "VC/VS", "VC", "VS", "AIC", "ALIC", "BNST"]:
        right_lead_location = "VC/VS"

    data = js["DiagnosticData"]["LFPTrendLogs"]
    data_left, data_right = [], []

    if "HemisphereLocationDef.Left" in data:
        for key in data["HemisphereLocationDef.Left"]:
            data_left.extend(data["HemisphereLocationDef.Left"][key])

    if "HemisphereLocationDef.Right" in data:
        for key in data["HemisphereLocationDef.Right"]:
            data_right.extend(data["HemisphereLocationDef.Right"][key])

    if len(data_left) > 0:
        left_timestamp, left_lfp, left_stim = map(
            list,
            zip(
                *(
                    (d["DateTime"], d["LFP"], d["AmplitudeInMilliAmps"])
                    for d in data_left
                )
            ),
        )
    if len(data_right) > 0:
        right_timestamp, right_lfp, right_stim = map(
            list,
            zip(
                *(
                    (d["DateTime"], d["LFP"], d["AmplitudeInMilliAmps"])
                    for d in data_right
                )
            ),
        )

    if len(data_left) > 0:
        df_left = pd.DataFrame(
            {
                "timestamp": left_timestamp,
                "lfp_left": left_lfp,
                "stim_left": left_stim,
            }
        )
    else:
        df_left = pd.DataFrame(
            {"timestamp": right_timestamp, "lfp_left": np.nan, "stim_left": np.nan}
        )
    if len(data_right) > 0:
        df_right = pd.DataFrame(
            {
                "timestamp": right_timestamp,
                "lfp_right": right_lfp,
                "stim_right": right_stim,
            }
        )
    else:
        df_right = pd.DataFrame(
            {"timestamp": left_timestamp, "lfp_right": np.nan, "stim_right": np.nan}
        )

    final_df = pd.merge(df_left, df_right, on="timestamp", how="outer")

    final_df["source_file"] = filename.name
    final_df["left_lead_location"] = left_lead_location
    final_df["right_lead_location"] = right_lead_location
    final_df["left_lead_model"] = left_lead_model
    final_df["right_lead_model"] = right_lead_model
    return final_df


def get_param_changes(js):
    """
    Extracts the changes in stimulation parameters from a list of JSON files.

    Parameters:
        jsons (list): List of JSON file paths.

    Returns:
        changes_dict (dict): Dictionary with timestamps as keys and changes in parameters as values.
    """

    params = {}
    groups = js["Groups"]
    for phase in ["Initial", "Final"]:
        try:
            active_group = next((d for d in groups[phase] if d["ActiveGroup"]), None)
            sensing_channel = active_group["ProgramSettings"]["SensingChannel"]
        except (KeyError, TypeError):
            continue
        phase_params = {}
        for hem in ["Left", "Right"]:
            hem_channel = next(
                (
                    d
                    for d in sensing_channel
                    if d["HemisphereLocation"] == f"HemisphereLocationDef.{hem}"
                ),
                None,
            )
            if hem_channel is None:
                continue
            try:
                contacts = [
                    d["Electrode"]
                    for d in hem_channel["ElectrodeState"]
                    if d["Electrode"] != "ElectrodeDef.Case"
                ]
            except KeyError:
                contacts = None
            try:
                pulse_width = hem_channel["PulseWidthInMicroSecond"]
            except KeyError:
                pulse_width = None
            try:
                rate = hem_channel["RateInHertz"]
            except KeyError:
                rate = None
            try:
                amplitude = hem_channel["SuspendAmplitudeInMilliAmps"]
            except KeyError:
                amplitude = None
            try:
                sensing_freq = hem_channel["SensingSetup"]["FrequencyInHertz"]
            except KeyError:
                sensing_freq = None
            hem_params = {
                "contacts": contacts,
                "pulse_width": pulse_width,
                "freq": rate,
                "amplitude": amplitude,
                "sense_freq": sensing_freq,
            }
            phase_params[hem] = hem_params
        params[phase] = phase_params
    if "Initial" not in params or "Final" not in params:
        return None
    if params["Initial"] != params["Final"]:
        common_keys = set(params["Initial"].keys()) & set(params["Final"].keys())
        if {k: params["Initial"][k] for k in common_keys} == {
            k: params["Final"][k] for k in common_keys
        }:
            return None
        try:
            timestamp = datetime.fromisoformat(
                js["SessionEndDate"].replace("Z", "+00:00")
            )
        except ValueError:
            try:
                timestamp = datetime.fromisoformat(
                    js["SessionDate"].replace("Z", "+00:00")
                )
            except ValueError:
                return None

        changes = {}
        for hem in ["Left", "Right"]:
            initial = params["Initial"].get(hem, {})
            final = params["Final"].get(hem, {})
            changes[hem] = {
                key: (initial.get(key), final.get(key))
                for key in (set(initial.keys()) | set(final.keys()))
                if initial.get(key) != final.get(key)
            }
        return {timestamp: changes}
    else:
        return None
