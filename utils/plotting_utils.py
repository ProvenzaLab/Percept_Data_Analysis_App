import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils.utils as utils
import scipy.stats as stats

# Color and style settings
C_PRE_DBS = "rgba(255, 215, 0, 0.5)"
C_REPSONDER = "rgba(0, 0, 255, 1)"
C_NON_RESPONDER = "rgba(255, 185, 0, 1)"
C_DISINHIBITED = "#ff0000"
C_DOTS = "rgba(128, 128, 128, 0.5)"
C_PRED = "rgba(51, 160, 44, 1)"
C_RAW = "rgba(128, 128, 128, 0.7)"


def plot_metrics(
    df: pd.DataFrame,
    patient: str,
    hemisphere: str,
    changes_df: pd.DataFrame,
    show_changes: bool,
    patients_dict: dict,
    param_dict: dict,
) -> go.Figure:
    """
    Generate a plot with multiple subplots to visualize various metrics including LFP amplitude, linear AR model,
    and R² values over time, before and after DBS.

    Parameters:
        df (pd.Dataframe): Dataframe containing processed percept data.
        patient (str): Patient identifier.
        hemisphere (int): Hemisphere index (0 or 1).
        changes_df (pd.Dataframe): Dataframe containing stimulation parameters and changes.

    Returns:
        go.Figure: A Plotly figure with the generated subplots.
    """

    patient_dict = patients_dict[patient]
    model = param_dict["model"]
    delta = param_dict["delta"]

    pt_df = df.query('pt_id == @patient and lead_location == "VC/VS"')
    pt_daily_df = pt_df.drop_duplicates(subset=["days_since_dbs"])

    days = pt_daily_df["days_since_dbs"]
    linAR_R2 = pt_daily_df[f"lfp_{hemisphere}_day_r2_{model}"]

    if delta:
        if pt_df.query("days_since_dbs < 0").empty:
            pass
        pt_daily_df[f"lfp_{hemisphere}_day_r2_{model}"] -= np.nanmean(
            pt_daily_df.query("days_since_dbs < 0")[f"lfp_{hemisphere}_day_r2_{model}"]
        )
        linAR_R2 = pt_daily_df[f"lfp_{hemisphere}_day_r2_{model}"]

    # Identify discontinuities in the days array
    start_index = np.where(np.diff(days) > 7)[0] + 1
    start_index = np.concatenate(([0], start_index, [len(days)]))

    # Create figure with subplots
    fig = make_subplots(
        rows=2,
        cols=4,
        row_heights=[0.5, 0.5],
        column_widths=[0.3, 0.35, 0.35, 0.1],
        specs=[
            [{"colspan": 4}, None, None, None],
            [{"colspan": 3}, None, None, {"colspan": 1}],
        ],
        subplot_titles=(
            "Full Time-Domain Plot",
            "Linear AR R² Over Time",
            "Linear AR R² Violin Plot",
        ),
    )

    # Set plot aesthetics
    grid_color = "#a0a0a0"
    title_font_color = "#2e2e2e"
    axis_title_font_color = "#2e2e2e"
    axis_line_color = "#2e2e2e"
    plot_bgcolor = "rgba(240, 240, 240, 1)"
    paper_bgcolor = "rgba(240, 240, 240, 1)"

    fig.update_layout(
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        annotations=[
            dict(
                text="",
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                showarrow=False,
                font=dict(size=20, color=title_font_color, family="Helvetica"),
            )
        ],
    )

    # Plot Full Time-Domain Plot
    for i in range(len(start_index) - 1):
        segment_days = np.ravel(days[start_index[i] + 1 : start_index[i + 1]])
        segment_df = pt_df.query("days_since_dbs in @segment_days")
        segment_OG = segment_df[f"lfp_{hemisphere}_z_scored_{model}"]
        segment_linAR = segment_df[f"lfp_{hemisphere}_preds_{model}"]
        segment_times = segment_df["CT_timestamp"]

        mask = ~np.isnan(segment_times) & ~np.isnan(segment_OG)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) > 0:
            segments = np.split(
                valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1
            )

            for segment in segments:
                if len(segment) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=segment_times.values[segment],
                            y=segment_OG.values[segment],
                            mode="lines",
                            line=dict(color=C_RAW, width=1),
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

        linAR_mask = ~np.isnan(segment_times) & ~np.isnan(segment_linAR)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) > 0:
            segments = np.split(
                valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1
            )

            for segment in segments:
                if len(segment) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=segment_times.values[segment],
                            y=segment_linAR.values[segment],
                            mode="lines",
                            line=dict(color=C_PRED, width=1),
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )
    fig.add_vline(
        x=patient_dict["dbs_date"],
        row=1,
        col=1,
        line_dash="dash",
        line_color="hotpink",
        line_width=5,
    )

    def get_change(values):
        if isinstance(values[0], (int, float)) and isinstance(values[1], (int, float)):
            return f"{values[0]} -> {values[1]}"
        elif isinstance(values[0], (int, float)) and values[1] is None:
            return f"{values[0]} -> Off"
        elif values[0] is None and isinstance(values[1], (int, float)):
            return f"Off -> {values[1]}"
        else:
            return f"Contact {values[0][0].split('_')[-1][0] if values[0] is not None else 'Off'} -> Contact {values[1][0].split('_')[-1][0] if values[1] is not None else 'Off'}"

    if show_changes:
        pt_changes = changes_df.query("pt_id == @patient")
        for i, row in pt_changes.iterrows():
            row.dropna(inplace=True)
            if (
                row.days_since_dbs == 0
                or pt_df.query("days_since_dbs == @row.days_since_dbs")[
                    f"stim_{hemisphere}"
                ]
                .isna()
                .all()
            ):
                continue
            changes = row.drop(
                labels=[label for label in row.index if hemisphere not in label]
            )
            annotation = [
                f"{" ".join(changes.index[i].split('_')[1:])}: {get_change(changes.values[i])}"
                for i in range(len(changes))
            ]

            ts = pd.to_datetime(row["CT_timestamp"], errors="coerce")

            if not pd.notnull(ts):
                continue

            dt = ts.to_pydatetime()
            fig.add_trace(
                go.Scatter(
                    x=[dt, dt],
                    y=[-3, 10],
                    mode="lines",
                    line=dict(color="black", width=3, dash="dot"),
                    hovertext=f"{ts}: {", ".join(annotation)}",
                    hoverinfo="text",
                    xaxis="x",
                    yaxis="y",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    # fig.add_trace(go.Scatter(x=linAR_t, y=pt_df.dropna(subset=[f'lfp_{hemisphere}_preds_{model}'])[f'lfp_{hemisphere}_preds_{model}'], mode='lines', name="Linear AR", line=dict(color=c_linAR, width=1.5), showlegend=False), row=1, col=1)
    fig.update_yaxes(
        title_text="LFP (z-scored)",
        row=1,
        col=1,
        tickfont=dict(color=axis_title_font_color),
        titlefont=dict(color=axis_title_font_color),
        showline=True,
        linecolor=axis_line_color,
    )
    fig.update_xaxes(
        title_text="Date (CT)",
        tickfont=dict(color=axis_title_font_color),
        titlefont=dict(color=axis_title_font_color),
        showline=True,
        linecolor=axis_line_color,
    )

    for i in range(len(start_index) - 1):
        segment_days = days.values[start_index[i] + 1 : start_index[i + 1]]
        segment_linAR_R2 = linAR_R2.values[start_index[i] + 1 : start_index[i + 1]]

        # Plot the dots
        fig.add_trace(
            go.Scatter(
                x=segment_days,
                y=pd.Series(segment_linAR_R2).rolling(window=5, min_periods=1).mean(),
                mode="lines",
                line=dict(color=C_DOTS),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    color_dict = {
        0: C_PRE_DBS,
        1: C_DISINHIBITED,
        2: C_NON_RESPONDER,
        3: C_REPSONDER,
        4: C_DOTS,
    }

    for color in color_dict.keys():
        if color not in pt_df["state_label"].values:
            continue
        state_df = pt_daily_df.query("state_label == @color")
        state_days = state_df["days_since_dbs"]
        state_r2 = state_df[f"lfp_{hemisphere}_day_r2_{model}"]

        fig.add_trace(
            go.Scatter(
                x=state_days,
                y=state_r2,
                mode="markers",
                line=dict(color=color_dict[color]),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.add_vline(
        x=0, row=2, col=1, line_dash="dash", line_color="hotpink", line_width=5
    )

    if show_changes:
        for i, row in pt_changes.iterrows():
            row.dropna(inplace=True)

            day = row["days_since_dbs"]
            if (
                day == 0
                or pt_daily_df.query("days_since_dbs == @day")[
                    f"lfp_{hemisphere}_day_r2_{model}"
                ]
                .isna()
                .all()
            ):
                continue
            fig.add_vline(
                day, row=2, col=1, line_dash="dot", line_color="black", line_width=3
            )

    fig.update_yaxes(
        title_text="Linear AR R²",
        row=2,
        col=1,
        tickfont=dict(color=axis_title_font_color),
        titlefont=dict(color=axis_title_font_color),
        showline=True,
        linecolor=axis_line_color,
    )
    fig.update_xaxes(
        title_text="Days Since DBS Activation",
        tickfont=dict(color=axis_title_font_color),
        titlefont=dict(color=axis_title_font_color),
        showline=True,
        linecolor=axis_line_color,
    )

    # Linear AR R² Violin Plot
    VIOLIN_WIDTH = 7.0
    violin_df = pt_daily_df.dropna(subset=[f"lfp_{hemisphere}_day_r2_{model}"])
    fig.add_trace(
        go.Violin(
            y=violin_df.query("state_label == 0")[f"lfp_{hemisphere}_day_r2_{model}"],
            name="",
            side="negative",
            line_color=C_PRE_DBS,
            fillcolor=C_PRE_DBS,
            showlegend=False,
            width=VIOLIN_WIDTH,
            meanline_visible=True,
            meanline=dict(color="black", width=2),
        ),
        row=2,
        col=4,
    )

    if patient_dict["response_status"] == 1:
        t_val, p_val = stats.ttest_ind(
            violin_df.query("state_label == 0")[f"lfp_{hemisphere}_day_r2_{model}"],
            violin_df.query("state_label == 3")[
                f"lfp_{hemisphere}_day_r2_{model}"
            ],
            equal_var=False,
        )
        fig.add_trace(
            go.Violin(
                y=violin_df.query("state_label == 3")[
                    f"lfp_{hemisphere}_day_r2_{model}"
                ],
                side="positive",
                line_color=C_REPSONDER,
                fillcolor=C_REPSONDER,
                showlegend=False,
                width=VIOLIN_WIDTH,
                meanline_visible=True,
                meanline=dict(color="white", width=2),
            ),
            row=2,
            col=4,
        )

    else:
        t_val, p_val = stats.ttest_ind(
            violin_df.query("state_label == 0")[f"lfp_{hemisphere}_day_r2_{model}"],
            violin_df.query("days_since_dbs > 0")[f"lfp_{hemisphere}_day_r2_{model}"],
            equal_var=False,
        )
        fig.add_trace(
            go.Violin(
                y=violin_df.query("days_since_dbs > 0")[
                    f"lfp_{hemisphere}_day_r2_{model}"
                ],
                side="positive",
                line_color=C_NON_RESPONDER,
                fillcolor=C_NON_RESPONDER,
                showlegend=False,
                width=VIOLIN_WIDTH,
                meanline_visible=True,
                meanline=dict(color="black", width=2),
            ),
            row=2,
            col=4,
        )

    fig.update_yaxes(
        row=2,
        col=4,
        tickfont=dict(color=axis_title_font_color),
        titlefont=dict(color=axis_title_font_color),
        showline=True,
        linecolor=axis_line_color,
    )
    fig.update_xaxes(
        title_text=f"t = {np.round(t_val, 3)}",
        row=2,
        col=4,
        tickfont=dict(color=axis_title_font_color),
        titlefont=dict(color=axis_title_font_color),
        showline=True,
        linecolor=axis_line_color,
    )
    # Set overall layout aesthetics
    fig.update_layout(
        height=650,
        width=900,
        showlegend=True,
        legend=dict(
            x=0.85,
            y=0.85,
            bgcolor="rgba(255, 255, 255, 0.7)",
            font=dict(color=title_font_color),
            itemsizing="constant",
            itemwidth=30,
        ),
        margin=dict(l=50, r=50, b=50, t=50),
        font=dict(color=title_font_color, family="Helvetica"),
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor=grid_color),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor=grid_color),
    )

    annotations = [
        dict(
            text="Full Time-Domain Plot",
            x=0.5,
            xref="paper",
            y=1,
            yref="paper",
            showarrow=False,
            font=dict(size=14, color=title_font_color, family="Helvetica"),
        ),
        dict(
            text="Linear AR R² Over Time",
            x=0.45,
            xref="paper",
            y=0.45,  # Adjusted Y position to move the annotation upwards
            yref="paper",
            showarrow=False,
            font=dict(size=14, color=title_font_color, family="Helvetica"),
        ),
        dict(
            text=utils.get_sig_text(p_val),
            x=0.965,
            xref="paper",
            y=0.3 if delta else 0.35,
            yref="paper",
            showarrow=False,
            font=dict(size=14, color=title_font_color, family="Helvetica"),
        ),
    ]

    fig.update_layout(
        annotations=annotations,
        legend=dict(
            x=1,
            y=0.5,
            xanchor="right",
            yanchor="middle",
            font=dict(size=10),
            entrywidth=0.5,
            entrywidthmode="fraction",
            itemwidth=30,
        ),
    )

    return fig, t_val, p_val


def make_legend(patient, patient_dict, show_changes):
    fig = make_subplots()
    pt_params = patient_dict[patient]

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Raw LFP (z-scored)",
            marker=dict(color=C_RAW, symbol="circle"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="AR(1) predicted LFP (z-scored)",
            marker=dict(color=C_PRED, symbol="circle"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Pre-DBS",
            marker=dict(color=C_PRE_DBS, symbol="circle"),
        )
    )
    if pt_params["response_status"] == 1:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Response",
                marker=dict(color=C_REPSONDER, symbol="circle"),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Non-response",
                marker=dict(color=C_NON_RESPONDER, symbol="circle"),
            )
        )
    if "disinhibited_dates" in list(pt_params.keys()):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Disinhibited",
                marker=dict(color=C_DISINHIBITED, symbol="circle"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="DBS On",
            marker=dict(color="hotpink", symbol="square"),
        )
    )
    if show_changes:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Paramter Change",
                marker=dict(color="black", symbol="square"),
            )
        )

    fig.update_layout(
        legend=dict(
            x=0,
            y=0,
            xanchor="center",
            yanchor="middle",
            font=dict(size=10),
            entrywidth=0.5,
            entrywidthmode="fraction",
            itemwidth=30,
        )
    )
    return fig
