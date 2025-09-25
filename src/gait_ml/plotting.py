import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_xyz(x, y, z, labels=None, title=None, start_list=None, end_list=None):
    assert len(x) == len(y)
    assert len(y) == len(z)
    time = np.arange(len(x))

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for each axis
    fig.add_trace(go.Scatter(x=time, y=x, mode="lines", name="X-axis"))
    fig.add_trace(go.Scatter(x=time, y=y, mode="lines", name="Y-axis"))
    fig.add_trace(go.Scatter(x=time, y=z, mode="lines", name="Z-axis"))

    # if labels:
    colors = [
        "#000000",
        "#CE4848",  # muted blue (Plotly default)
        "#4f96dd",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#4f96dd",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
    ]
    if labels is not None:
        for k, v in enumerate(labels):
            if v == 1 or v == 2:
                fig.add_vline(x=k, line_color=colors[v], line_dash="dash")

    if start_list is not None and end_list is not None:
        for start, end in zip(start_list, end_list):
            fig.add_vrect(
                x0=start, x1=end, fillcolor="#de2828", opacity=0.2, line_width=2
            )

    # Update layout for title and labels
    fig.update_layout(
        title=title,
        xaxis_title="Samples",
        yaxis_title="Unit",
        hovermode="x unified",
        height=500,
    )

    return fig


def plot_3d_points_from_df(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    color_col: str = None,
    size_col: str = None,
    hover_name_col: str = None,
    title: str = "3D Scatter Plot of Points",
) -> go.Figure:
    """
    Creates an interactive 3D scatter plot from a Pandas DataFrame using Plotly Express.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the 3D point data.
    x_col : str, optional
        The name of the column in `df` that contains the x-coordinates. Defaults to 'x'.
    y_col : str, optional
        The name of the column in `df` that contains the y-coordinates. Defaults to 'y'.
    z_col : str, optional
        The name of the column in `df` that contains the z-coordinates. Defaults to 'z'.
    color_col : str, optional
        The name of a column in `df` to use for coloring the points. If None, all points
        will have the same color. Defaults to None.
    size_col : str, optional
        The name of a column in `df` to use for sizing the points. If None, all points
        will have the same size. Defaults to None.
    hover_name_col : str, optional
        The name of a column in `df` to use for the hover text (displayed when hovering
        over a point). Defaults to None.
    title : str, optional
        The title of the plot. Defaults to "3D Scatter Plot of Points".

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly Figure object that can be displayed using fig.show().

    Raises
    ------
    ValueError
        If any of the specified x_col, y_col, or z_col do not exist in the DataFrame.
    """
    required_cols = [x_col, y_col, z_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the DataFrame.")

    # Create the 3D scatter plot using plotly.express
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        size=size_col,
        hover_name=hover_name_col,
        title=title,
        height=700,  # Adjust height for better visualization
    )

    # Customize the layout for better readability and interaction
    fig.update_layout(
        scene=dict(
            xaxis_title=f"{x_col} Axis",
            yaxis_title=f"{y_col} Axis",
            zaxis_title=f"{z_col} Axis",
            bgcolor="white",  # Set background color
            xaxis=dict(backgroundcolor="lightgrey", gridcolor="white"),
            yaxis=dict(backgroundcolor="lightgrey", gridcolor="white"),
            zaxis=dict(backgroundcolor="lightgrey", gridcolor="white"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margins
        hovermode="closest",  # Improve hover behavior
    )

    return fig
