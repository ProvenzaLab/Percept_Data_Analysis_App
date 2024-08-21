import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate the weighted median of the given values with the specified weights.

    Parameters:
        values (np.ndarray): Array of values.
        weights (np.ndarray): Array of weights corresponding to the values.

    Returns:
        float: The weighted median of the values.
    """
    sorted_indices = np.argsort(values)
    sorted_data = np.array(values)[sorted_indices]
    sorted_weights = np.array(weights)[sorted_indices]
    cumulative_weight = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumulative_weight, 0.5 * cumulative_weight[-1])
    return sorted_data[median_idx]

def _emm_plot(days: np.ndarray, stat: np.ndarray, ema_skip: int, span: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Exponential Moving Median (EMM) for the given statistical data, skipping a specified number of points.

    Parameters:
        days (np.ndarray): Array of days corresponding to the data.
        stat (np.ndarray): Statistical data to calculate the EMM on.
        ema_skip (int): Number of points to skip at the beginning of the EMM calculation.
        span (int): The span or window size for the EMM calculation. Default is 5.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of days and corresponding EMM values, starting after the skipped points.
    """
    # Ensure the input is a numpy array for consistent behavior
    stat = np.array(stat)

    # Find the index to start the calculation, skipping any NaNs and the specified number of points
    skip_idx = max(ema_skip, np.where(~np.isnan(stat))[0][0])
    
    # Calculate weights for the EMM
    weights = np.exp(np.linspace(-1., 0., span))
    weights /= weights.sum()

    # Initialize the EMM values
    emm_values = []

    for i in range(skip_idx, len(stat)):
        if i < span:
            # If we don't have enough data points for the full span, skip this index
            emm_values.append(np.nan)
        else:
            window = stat[i-span+1:i+1]
            if np.any(np.isnan(window)):
                emm_values.append(np.nan)
            else:
                median_value = _weighted_median(window, weights)
                emm_values.append(median_value)

    return days[skip_idx:], np.array(emm_values)

def _ema_plot(days: np.ndarray, stat: np.ndarray, ema_skip: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Exponential Moving Average (EMA) for the given statistical data, skipping a specified number of points.

    Parameters:
        days (np.ndarray): Array of days corresponding to the data.
        stat (np.ndarray): Statistical data to calculate the EMA on.
        ema_skip (int): Number of points to skip at the beginning of the EMA calculation.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of days and corresponding EMA values, starting after the skipped points.
    """
    skip_idx = max(ema_skip, np.where(~np.isnan(stat))[0][0])
    ema = pd.Series(stat).ewm(span=5, adjust=False).mean().fillna(0).to_numpy()
    return days[skip_idx:], ema[skip_idx:]

def plot_metrics(
    percept_data: dict, 
    subject: str, 
    hemisphere: int, 
    pre_DBS_bounds: tuple[int, int], 
    post_DBS_bounds: tuple[int, int], 
    zone_index: dict
) -> go.Figure:
    """
    Generate a plot with multiple subplots to visualize various metrics including LFP amplitude, linear AR model,
    and R² values over time, before and after DBS.

    Parameters:
        percept_data (dict): Dictionary containing processed percept data.
        subject (str): Subject identifier.
        hemisphere (int): Hemisphere index (0 or 1).
        pre_DBS_bounds (tuple[int, int]): X-axis bounds for the pre-DBS zoomed plot.
        post_DBS_bounds (tuple[int, int]): X-axis bounds for the post-DBS zoomed plot.
        zone_index (dict): Dictionary containing responder and non-responder indices for each subject.

    Returns:
        go.Figure: A Plotly figure with the generated subplots.
    """
    # Color and style settings
    c_preDBS = 'rgba(255, 215, 0, 0.5)'
    c_linAR = 'rgba(51, 160, 44, 1)'
    c_dots = 'rgba(128, 128, 128, 0.5)'
    c_OG = 'rgba(128, 128, 128, 0.7)'
    sz = 5
    ylim_LFP = [-2, 6]
    ylim_R2 = [-49, 90]
    ema_skip = 3

    # Extract relevant data
    patient_idx = subject
    days = percept_data['days'][patient_idx][hemisphere]
    days_OG = days.copy()
    t = percept_data['time_matrix'][patient_idx][hemisphere]
    OG = percept_data['LFP_filled_matrix'][patient_idx][hemisphere]
    linAR = percept_data['linearAR_matrix'][patient_idx][hemisphere]
    linAR_R2 = percept_data['linearAR_R2'][patient_idx][hemisphere]

    # Identify responder and non-responder indices
    pre_DBS_idx = np.where(days < 0)[0]
    try:
        responder_idx = np.intersect1d(days, zone_index['responder'][patient_idx], return_indices=True)[1]
        non_responder_idx = np.intersect1d(days, zone_index['non_responder'][patient_idx], return_indices=True)[1]
    except:
        responder_idx = np.asarray([], dtype=int)
        non_responder_idx = np.asarray([], dtype=int)

    # Identify discontinuities in the days array
    start_index = np.where(np.diff(days) > 7)[0] + 1
    start_index = np.concatenate(([0], start_index, [len(days)]))

    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=4,
        row_heights=[0.2, 0.3, 0.5],
        column_widths=[0.3, 0.35, 0.35, 0.1],
        specs=[[{"colspan": 4}, None, None, None],
               [{"colspan": 2}, None, {"colspan": 2}, None],
               [{"colspan": 3}, None, None, {"colspan": 1}]],
        subplot_titles=("Full Time-Domain Plot", "Zoomed Pre-DBS", "Zoomed Post-DBS",
                        "Linear AR R² Over Time", "Linear AR R² Violin Plot"))

    # Set plot aesthetics
    grid_color = '#a0a0a0'
    title_font_color = '#2e2e2e'
    axis_title_font_color = '#2e2e2e'
    axis_line_color = '#2e2e2e'
    plot_bgcolor = 'rgba(240, 240, 240, 1)'
    paper_bgcolor = 'rgba(240, 240, 240, 1)'

    fig.update_layout(
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        annotations=[dict(
            text='',
            xref='paper',
            yref='paper',
            x=0,
            y=1,
            showarrow=False,
            font=dict(
                size=20,
                color=title_font_color,
                family="Helvetica"
            )
        )]
    )

    # Plot Full Time-Domain Plot
    for i in range(len(start_index) - 1):
        segment_t = np.ravel(t[:, start_index[i]+1:start_index[i+1]], order='F')
        segment_OG = np.ravel(OG[:, start_index[i]+1:start_index[i+1]], order='F')

        mask = ~np.isnan(segment_t) & ~np.isnan(segment_OG)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            segments = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
            
            for segment in segments:
                if len(segment) > 0:
                    fig.add_trace(go.Scatter(
                        x=segment_t[segment],
                        y=segment_OG[segment],
                        mode='lines',
                        line=dict(color=c_OG, width=1),
                        showlegend=False
                    ), row=1, col=1)
    fig.update_yaxes(title_text="9 Hz LFP (mV)", row=1, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)


    t_zoom = np.ravel(t, order='F')[~np.isnan(np.ravel(t, order='F'))]
    OG_zoom = np.ravel(OG, order='F')[~np.isnan(np.ravel(OG, order='F'))]
    linAR_zoom = np.ravel(linAR, order='F')[~np.isnan(np.ravel(linAR, order='F'))]
    
    # Zoomed Pre-DBS
    fig.add_trace(go.Scatter(x=t_zoom, y=OG_zoom, mode='lines', name="Original", line=dict(color=c_OG, width=2), showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_zoom, y=linAR_zoom, mode='lines', name="Linear AR", line=dict(color=c_linAR, width=1.5), showlegend=True), row=2, col=1)
    fig.update_yaxes(title_text="9 Hz LFP Amplitude (mV)", range=ylim_LFP, row=2, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(range=pre_DBS_bounds, row=2, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)

    # Zoomed Post-DBS
    fig.add_trace(go.Scatter(x=t_zoom, y=OG_zoom, mode='lines', name="Original", line=dict(color=c_OG, width=2), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=t_zoom, y=linAR_zoom, mode='lines', name="Linear AR", line=dict(color=c_linAR, width=1.5), showlegend=False), row=2, col=3)
    fig.update_yaxes(range=ylim_LFP, row=2, col=3, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(range=post_DBS_bounds, row=2, col=3, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)

    # Linear AR R² Over Time
    for i in range(len(start_index) - 1):
        fig.add_trace(go.Scatter(x=days[start_index[i]+1:start_index[i+1]], 
                                 y=linAR_R2[start_index[i]+1:start_index[i+1]]*100, 
                                 mode='markers', marker=dict(color=c_dots, size=sz), showlegend=False),
                      row=3, col=1)
        days_ema, linAR_R2_ema = _emm_plot(days[start_index[i]+1:start_index[i+1]], linAR_R2[start_index[i]+1:start_index[i+1]], ema_skip)
        fig.add_trace(go.Scatter(x=days_ema, y=linAR_R2_ema*100, mode='lines', line=dict(color=c_linAR), showlegend=False),
                      row=3, col=1)
    fig.update_yaxes(title_text="Linear AR R² (%)", range=ylim_R2, row=3, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)

    # Linear AR R² Violin Plot
    fig.add_trace(go.Violin(y=linAR_R2[days < 0]*100, name="Pre-DBS", side='negative', line_color=c_linAR, showlegend=False),
                  row=3, col=4)
    fig.update_yaxes(range=ylim_R2, row=3, col=4, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)

    # Set overall layout aesthetics
    fig.update_layout(
        height=650,
        width=900,
        showlegend=True,
        legend=dict(x=0.85, y=0.85, bgcolor='rgba(255, 255, 255, 0.7)', font=dict(color=title_font_color), itemsizing='constant', itemwidth=30),
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
            font=dict(size=14, color=title_font_color, family="Helvetica")
        ),
        dict(
            text="Zoomed Pre-DBS",
            x=0.2,
            xref="paper",
            y=0.75,
            yref="paper",
            showarrow=False,
            font=dict(size=14, color=title_font_color, family="Helvetica")
        ),
        dict(
            text="Zoomed Post-DBS",
            x=0.75,
            xref="paper",
            y=0.75,
            yref="paper",
            showarrow=False,
            font=dict(size=14, color=title_font_color, family="Helvetica")
        ),
        dict(
            text="Linear AR R² Over Time",
            x=0.35,
            xref="paper",
            y=0.35,  # Adjusted Y position to move the annotation upwards
            yref="paper",
            showarrow=False,
            font=dict(size=14, color=title_font_color, family="Helvetica")
        )
    ]

    fig.update_layout(annotations=annotations)

    return fig
