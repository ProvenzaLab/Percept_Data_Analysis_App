import numpy as np
import pandas as pd
from datetime import timedelta,date
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold  # type: ignore
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import (  # type: ignore
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    r2_score,
    accuracy_score,
)
from typing import Any, Dict, Tuple, Iterable
from statsmodels.tsa.stattools import acf
from scipy import stats


default_colors = np.array(["gold", "r", "orange", "b", "gray", "darkgray"]).astype(
    object
)


def predict_series_calc_R2(group: pd.DataFrame, ar_features: list, gt_colname: str, test_day: int,
                           new_cols: dict, window_size: int=3):
    """
    Fits autoregressive AR(1) model to data, then tests on a single day and returns predictions, daily R2 scores, prediction residuals, and residual variance.

    Parameters:
    - group (pd.DataFrame): A DataFrame containing processed data including state labels.
    - ar_features (list): A list of features to use for autoregression. If not None, this will be used instead of the num_lags parameter.
    - gt_colname (str): Name of the column for the feature the model is being trained to predict.
    - test_day (int): The day # (since DBS) to test the model on.
    - window_size (int): The size of the sliding window to use.

    Returns:
    - results_df (pd.DataFrame): A DataFrame containing predictions, daily R2 scores, prediction residuals, residual variance, and other spiking parameters.
    """

    all_days = group['days_since_dbs']
    unique_days = all_days.unique()
    
    test_df = group[all_days == test_day]
    train_df = group[all_days != test_day]

    results_df = pd.DataFrame(np.nan, index=test_df.index, columns=new_cols.values())
    
    if (unique_days[-1] - unique_days[0] != (window_size-1)) or (len(unique_days) != window_size): # Skip non-contiguous days
        return None
    
    train_df_no_na = train_df.dropna(subset=ar_features+[gt_colname])
    test_df_no_na = test_df.dropna(subset=ar_features+[gt_colname])
    if train_df_no_na.shape[0] < ((24 * 6) * (window_size - 2) + 1) or test_df_no_na.shape[0] < (24 * 6 // 2): # If we don't have enough data, just skip this day.
        return None

    if len(ar_features) > 1: # Select significant lags if using LinAR-k model
        sig_lags = ar_features.copy()
        sig_lags = select_lags_full_pipeline(train_df_no_na, ar_features, gt_colname, n_splits=5, fold_threshold=3)
        ar_features = sig_lags

    model = sm.OLS(train_df_no_na[gt_colname], train_df_no_na[ar_features]).fit()
    phi_1 = model.params[ar_features[0]]

    # Generate predictions for the test data using the fitted model.
    preds = model.predict(test_df_no_na[ar_features])

    # Save predictions to dataframe
    results_df.loc[test_df_no_na.index, new_cols['preds']] = preds.values

    # Save phi_1 to dataframe
    results_df.loc[test_df_no_na.index, new_cols['phi_1']] = phi_1

    # Calculate all features from the predictions
    residuals = results_df[new_cols['preds']] - test_df[gt_colname]
    abs_residuals = np.abs(residuals)
    metrics_to_compute = {
        'residuals': lambda: residuals,
        'R2': lambda: r2_score(
            test_df_no_na[gt_colname],
            results_df.loc[test_df_no_na.index, new_cols['preds']]
        ),
    }

    for name, func in metrics_to_compute.items():
        if name in new_cols:
            results_df[new_cols[name]] = func()
    return results_df

def apply_sliding_window(g, ar_features: list, gt_colname: str, window_size: int=3):
    '''
    Apply a sliding window to a groupby object and return a DataFrame with the results of the sliding window.

    Parameters:
    - g (pd.DataFrame): A DataFrame containing processed data including state labels from a single lead.
    - ar_features (list): A list of features to use for autoregression. If not None, this will be used instead of the num_lags parameter.
    - gt_colname (str): Name of the column for the feature the model is being trained to predict.
    - window_size (int): The size of the sliding window to use.
    '''
    g = g.dropna(subset=gt_colname).copy()

    new_cols = {
        'phi_1': gt_colname.replace('z_scored', 'phi_1'),
        'preds': gt_colname.replace('z_scored', 'preds'),
        'residuals': gt_colname.replace('z_scored', 'residuals'),
        'R2': gt_colname.replace('z_scored', 'day_r2'),
    }

    if g.empty:
        return None
    unique_dates = g['days_since_dbs'].unique()
    results = []

    for this_date in unique_dates:
        td_arr = np.arange(-window_size+1, 1)
        date_list = td_arr + this_date
        day_mask = g['days_since_dbs'].isin(date_list)

        window = g[day_mask]
        day_results = predict_series_calc_R2(window, ar_features, gt_colname, this_date, new_cols, window_size)
        if day_results is not None:
            results.append(day_results)
    return pd.concat(results) if len(results) > 0 else None

def select_significant_lags_kfold(df, lag_features, target_col, n_splits=5, p_thresh=0.05, threshold=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    pval_counts = {lag: 0 for lag in lag_features}

    for train_idx, _ in kf.split(df):
        train_data = df.iloc[train_idx]
        X = sm.add_constant(train_data[lag_features])
        y = train_data[target_col]
        model = sm.OLS(y, X).fit()
        for lag in lag_features:
            if lag in model.pvalues and model.pvalues[lag] < p_thresh:
                pval_counts[lag] += 1

    return [lag for lag, count in pval_counts.items() if count > threshold]


def select_lags_full_pipeline(df, lag_features, target_col, n_splits=5, p_thresh=0.05,
                               fold_threshold=3, max_iter=20, verbose=False):
    df = df.dropna(subset=lag_features + [target_col]).copy()
    candidates = list(lag_features)

    for i in range(max_iter):
        if not candidates:
            return [lag_features[0]] # Default to 1 lag term (AR-1)

        selected = select_significant_lags_kfold(
            df, candidates, target_col, n_splits, p_thresh, fold_threshold
        )
        if not selected:
            if verbose:
                print(f"Iteration {i}: no lags survived k-fold screening.")
            return [lag_features[0]] # Default to 1 lag term (AR-1)

        X_full = sm.add_constant(df[selected])
        full_model = sm.OLS(df[target_col], X_full).fit()
        non_sig = [lag for lag in selected if full_model.pvalues.get(lag, 1) >= p_thresh]

        if not non_sig:
            if verbose:
                print(f"Converged after {i+1} iteration(s): {selected}")
            return selected

        if verbose:
            print(f"Iteration {i}: dropping {non_sig} (non-significant on full data)")
        candidates = [lag for lag in selected if lag not in non_sig]

    if verbose:
        print("Max iterations reached without full convergence.")


    return candidates if len(candidates) > 0 else [lag_features[0]] # Default to 1 lag term (AR-1)

def leave_one_patient_out_logistic_regression(
    df: pd.DataFrame,
    feature_cols: Iterable,
    show_roc_curve: bool = True,
    show_conf_mat: bool = True,
    show_violin_plot: bool = True,
    colors: list = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform leave-one-patient-out cross-validation with logistic regression.

    Args:
        df (pd.DataFrame): Dataframe containing patient data.
        feature_cols (list): Column names to be used as feature.
        show_roc_curve (bool): Whether to plot ROC curve.
        show_conf_mat (bool): Whether to plot confusion matrix.
        show_violin_plot (bool): Whether to plot violin plot of predictions.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Overall results and patient-specific results.
    """

    if colors is None:
        colors = default_colors

    all_y_true, all_y_pred, all_y_prob = [], [], []
    pt_results_dict = {}
    patients = df["pt_id"].unique()

    for patient in patients:
        pt_results = {}

        # Split data
        train_df = df[df["pt_id"] != patient].copy()  # All other patients
        test_df = df[df["pt_id"] == patient].copy()  # Held-out patient

        # Filter out unknown and transition labels
        bad_labels = ["Unknown", "Transition"]
        train_df.query(f"state_label_str not in @bad_labels", inplace=True)
        test_df.query(f"state_label_str not in @bad_labels", inplace=True)

        # Map state_label: 0,2 → 0 (non-response), 3 → 1 (response)
        train_df["label"] = train_df["state_label"].map({0: 0, 2: 0, 3: 1})
        test_df["label"] = test_df["state_label"].map({0: 0, 2: 0, 3: 1})

        # Prepare features and labels
        X_train, y_train = train_df[feature_cols], train_df["label"]
        X_test, y_test = test_df[feature_cols], test_df["label"]

        # Skip if there's only one class in training data
        if len(y_train.unique()) < 2:
            print(
                f"Skipping patient {patient} due to insufficient class diversity in training."
            )
            continue

        # Skip if patient doesn't have any data in the test set
        if test_df.shape[0] < 2:
            continue

        # Train logistic regression
        model = LogisticRegression(class_weight="balanced", penalty=None).fit(
            X_train, y_train
        )

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[
            :, 1
        ]  # Probability for class 1 (responders)

        # Store results
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        pt_results["confusion_matrix"] = confusion_matrix(y_test, y_pred)
        pt_results["y_true"] = y_test.values
        pt_results["y_pred"] = y_pred
        pt_results["y_prob"] = y_prob
        pt_results["auc"] = (
            roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan
        )
        pt_results["balanced_accuracy"] = (
            balanced_accuracy_score(y_test, y_pred)
            if len(np.unique(y_test)) > 1
            else np.nan
        )
        pt_results["raw_accuracy"] = accuracy_score(y_test, y_pred)
        pt_results["model"] = model
        pt_results_dict[patient] = pt_results

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    # Compute overall confusion matrix
    conf_matrix = confusion_matrix(all_y_true, all_y_pred)

    # Compute overall AUC and balanced accuracy
    auc_score = (
        roc_auc_score(all_y_true, all_y_prob)
        if len(np.unique(all_y_true)) > 1
        else np.nan
    )
    balanced_acc = balanced_accuracy_score(all_y_true, all_y_pred)

    # Compute overall ROC curve
    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)

    if show_roc_curve:
        # Plot overall ROC curve
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f"Overall ROC (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"({feature_cols})\nOverall ROC Curve for LOPO CV")
        plt.legend()
        plt.gca().set(xlim=[0, 1.01], ylim=[0, 1.01])
        plt.show()

    if show_conf_mat:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix, display_labels=["Non-Responder", "Responder"]
        )
        disp.plot()
        plt.title(f"({feature_cols})\nOverall ROC Curve for LOPO CV")
        plt.show()

    if show_violin_plot:
        if len(feature_cols) != 1:
            print(
                f"Feature columns should be a single column for violin plot. Only using first feature column: {feature_cols[0]}"
            )
        feature_col = feature_cols[0]
        plt.figure(figsize=(6, 4))
        nr_data = df.query(
            'state_label_str == "Pre-DBS" or state_label_str == "Non-Responder"'
        )[feature_col]
        r_data = df.query('state_label_str == "Responder"')[feature_col]
        nr_parts = plt.violinplot(nr_data, positions=[1], showextrema=False, side="low")
        r_parts = plt.violinplot(r_data, positions=[1], showextrema=False, side="high")

        nr_body = nr_parts["bodies"][0]
        nr_body.set_facecolor(colors[0])
        nr_body.set_edgecolor(colors[0])
        nr_body.set_alpha(0.5)
        r_body = r_parts["bodies"][0]
        r_body.set_facecolor(colors[3])
        r_body.set_edgecolor(colors[3])
        r_body.set_alpha(0.5)

        a = 0.1
        plt.scatter(
            np.array([0.7] * len(nr_data))
            + np.random.RandomState(42).normal(0, 0.01, size=len(nr_data)),
            nr_data,
            c=colors[0],
            marker="o",
            s=15,
            alpha=a,
        )
        plt.scatter(
            np.array([1.3] * len(r_data))
            + np.random.RandomState(42).normal(0, 0.01, size=len(r_data)),
            r_data,
            c=colors[3],
            marker="o",
            s=15,
            alpha=a,
        )

        plt.gca().set(
            xlim=[0.6, 1.4],
            xticks=[0.8, 1.2],
            xticklabels=["Symptomatic State", "Response State"],
            ylabel=f"{feature_cols}",
            title=f"AUC Score: {auc_score:.4f}\n"
            + f"Balanced Accuracy: {balanced_acc:.4f}",
        )
        plt.gca().tick_params("x", tick1On=False, tick2On=False)

        plt.show()

    # Return aggregated results
    return {
        "confusion_matrix": conf_matrix,
        "AUC": auc_score,
        "balanced_accuracy": balanced_acc,
    }, pt_results_dict


def effective_sample_size(data, max_lag=40):
    """
    Calclulates the effective sample size (ESS) of a time series using the autocorrelation function (ACF).

    Parameters:
    - data (np.ndarray): The time series data.
    - max_lag (int): The maximum lag to consider for the ACF.

    Returns:
    - ess (float): The effective sample size.
    """
    acf_vals = acf(data, nlags=max_lag, fft=True)
    ess = len(data) / (1 + 2 * np.sum(acf_vals[1:]))
    return ess


def get_t_test_p_value(data1, data2, apply_ess=True, max_lag=40):
    """
    Calculates the p-value for a two-sample t-test between two sets of data.

    Parameters:
    - data1 (np.ndarray): The first set of data.
    - data2 (np.ndarray): The second set of data.
    - apply_ess (bool): Whether to apply effective sample size (ESS) adjustment.
    - max_lag (int): The maximum lag to consider for the ACF if apply_ess is True.

    Returns:
    - p_val (float): The p-value from the t-test.
    """
    # Calculate ESS for the two sets
    if apply_ess:
        ess1 = effective_sample_size(data1, max_lag=max_lag)
        ess2 = effective_sample_size(data2, max_lag=max_lag)
    else:
        ess1 = len(data1)
        ess2 = len(data2)

    # Calculate means and variances
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)

    # Adjusted standard error using ESS
    se = np.sqrt(var1 / ess1 + var2 / ess2)

    # T-statistic
    t_stat = (mean1 - mean2) / se

    # Degrees of freedom (Welch-Satterthwaite)
    df = (var1 / ess1 + var2 / ess2) ** 2 / (
        (var1**2) / (ess1**2 * (ess1 - 1)) + (var2**2) / (ess2**2 * (ess2 - 1))
    )

    # P-value (two-sided)
    p_val = 2 * stats.t.sf(np.abs(t_stat), df)

    return p_val
