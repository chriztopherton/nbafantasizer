"""
Styling module for Fantasy Points Analyzer.

This module contains all CSS styling, color gradient functions, and dataframe styling handlers.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def apply_custom_css():
    """
    Apply custom CSS styles to the Streamlit application.

    This function applies fantasy stat tracker look and feel styling including:
    - Header and typography styles
    - Sidebar styling
    - Button and tab styling
    - Dataframe styling
    - Info/alert box styling
    """
    st.markdown(
        """
    <style>
    /* Main styling to match fantasy stat tracker */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    h1 {
        color: #1A1A1A;
        font-weight: 600;
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    h2 {
        color: #1A1A1A;
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #FF6C37;
        padding-bottom: 0.5rem;
    }

    h3 {
        color: #333333;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 1rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }

    [data-testid="stSidebar"] [data-testid="stHeader"] {
        background-color: #FFFFFF;
        color: #FF6C37;
        font-weight: 600;
    }

    /* Section headers with orange accent */
    .section-header {
        background-color: #FF6C37;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #1A1A1A;
    }

    [data-testid="stMetricLabel"] {
        color: #666666;
    }

    /* Button styling */
    .stButton > button {
        background-color: #FF6C37;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 500;
        transition: background-color 0.3s;
    }

    .stButton > button:hover {
        background-color: #E55A2E;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #666666;
        font-weight: 500;
        padding: 10px 20px;
        border-radius: 4px 4px 0 0;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FF6C37;
        color: white;
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 4px;
        overflow: hidden;
    }

    /* Info/Alert boxes */
    .stInfo {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }

    .stSuccess {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
    }

    .stWarning {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
    }

    .stError {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
    }

    /* Sidebar selectbox styling */
    [data-testid="stSidebar"] .stSelectbox label {
        color: #333333;
        font-weight: 500;
    }

    /* Remove Streamlit branding - but keep header for sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hide only the Streamlit logo/branding text, not the sidebar toggle button */
    header[data-testid="stHeader"] a[href*="streamlit"],
    header[data-testid="stHeader"] img[alt*="Streamlit"],
    header[data-testid="stHeader"] img[src*="logo"] {
        display: none !important;
    }

    /* Ensure sidebar toggle button is always visible - target all header buttons */
    header[data-testid="stHeader"] button {
        visibility: visible !important;
        display: inline-flex !important;
        opacity: 1 !important;
    }

    /* Ensure sidebar is always visible and properly styled */
    section[data-testid="stSidebar"] {
        visibility: visible !important;
    }
    </style>
""",
        unsafe_allow_html=True,
    )


def color_text_gradient(series, cmap_name="Greens"):
    """
    Apply color gradient to text based on value in a Series.

    This function is used to style dataframe cells with color gradients,
    where higher values get darker colors from the specified colormap.

    Args:
        series: pandas Series with numeric values to colorize
        cmap_name: Name of matplotlib colormap to use (default: "Greens")

    Returns:
        pandas Series with CSS style strings for each cell
    """
    # Convert to numeric, handling errors
    numeric_series = pd.to_numeric(series, errors="coerce")

    # Get min and max for normalization
    min_val = numeric_series.min()
    max_val = numeric_series.max()

    # Handle edge cases
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(
            ["color: #1A1A1A; background-color: transparent;"] * len(series),
            index=series.index,
        )

    # Normalize values to 0-1 range
    normalized = (numeric_series - min_val) / (max_val - min_val)

    # Get color from colormap
    cmap = plt.cm.get_cmap(cmap_name)

    # Apply color to each value
    styles = []
    for norm_val in normalized:
        if pd.isna(norm_val):
            styles.append("color: #1A1A1A; background-color: transparent;")
        else:
            rgba = cmap(norm_val)
            hex_color = mcolors.rgb2hex(rgba[:3])
            styles.append(f"color: {hex_color}; background-color: transparent;")

    return pd.Series(styles, index=series.index)


def apply_game_log_styling(display_df, decimal_columns=None, highlight_live_row=False):
    """
    Apply styling to game log dataframe with color gradients.

    This function applies green color gradients to positive stat columns
    (FP, points, rebounds, assists, etc.) and red gradients to turnovers.
    Optionally highlights the first row (live stats) with a different background.

    Args:
        display_df: pandas DataFrame to style (game log data)
        decimal_columns: List of column names to format with 2 decimal places
        highlight_live_row: If True, highlights the first row with a red background

    Returns:
        Styled pandas DataFrame ready for display
    """
    # Update decimal_columns to use new column names
    if decimal_columns is None:
        decimal_columns = ["Fan Pts", "FP", "fieldGoals%", "freeThrows%"]
    if decimal_columns is None:
        decimal_columns = ["FP", "fieldGoals%", "freeThrows%"]

    # Columns to apply green gradient styling (updated for new format)
    columns_to_style = [
        "Fan Pts",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "ST",
        "BLK",
        # Legacy column names for backward compatibility
        "FP",
        "numMinutes",
        "points",
        "reboundsTotal",
        "assists",
        "steals",
        "blocks",
        "fieldGoals%",
        "freeThrows%",
    ]

    # Only style columns that exist in the dataframe
    columns_to_style = [col for col in columns_to_style if col in display_df.columns]

    # Format decimal columns to always show 2 decimal places
    format_dict = {}
    for col in decimal_columns:
        if col in display_df.columns:
            format_dict[col] = "{:.2f}"

    # Create styled dataframe with text color gradient
    styled_df = display_df.style.format(format_dict)

    # Highlight live stats row (first row) with red background and white text
    if highlight_live_row and len(display_df) > 0:
        # Apply background color to first row
        def highlight_live(row):
            if row.name == 0:  # First row
                return ["background-color: #FF4444; color: white; font-weight: bold;"] * len(row)
            return [""] * len(row)

        styled_df = styled_df.apply(highlight_live, axis=1)

    # Apply text color gradient to positive columns (darker greens for light theme)
    # Skip first row if it's live stats (already styled)
    for col in columns_to_style:
        if col in display_df.columns:
            if highlight_live_row and len(display_df) > 0:
                # Apply gradient only to rows after the first
                def gradient_with_skip(series):
                    result = pd.Series(index=series.index, dtype=object)
                    for idx in series.index:
                        if idx == 0 and highlight_live_row:
                            result[idx] = "color: white; background-color: transparent;"
                        else:
                            # Apply gradient for other rows
                            grad_series = color_text_gradient(series.loc[[idx]], cmap_name="Greens")
                            result[idx] = grad_series.iloc[0]
                    return result

                styled_df = styled_df.apply(
                    gradient_with_skip,
                    subset=[col],
                    axis=0,
                )
            else:
                styled_df = styled_df.apply(
                    lambda x: color_text_gradient(x, cmap_name="Greens"),
                    subset=[col],
                    axis=0,
                )

    # Apply red text color gradient to turnovers (skip first row if live)
    turnover_cols = [col for col in ["TO", "turnovers"] if col in display_df.columns]
    if turnover_cols:
        if highlight_live_row and len(display_df) > 0:

            def turnover_gradient_with_skip(series):
                result = pd.Series(index=series.index, dtype=object)
                for idx in series.index:
                    if idx == 0:
                        result[idx] = "color: white; background-color: transparent;"
                    else:
                        grad_series = color_text_gradient(series.loc[[idx]], cmap_name="Reds")
                        result[idx] = grad_series.iloc[0]
                return result

            styled_df = styled_df.apply(
                turnover_gradient_with_skip,
                subset=turnover_cols,
                axis=0,
            )
        else:
            styled_df = styled_df.apply(
                lambda x: color_text_gradient(x, cmap_name="Reds"),
                subset=turnover_cols,
                axis=0,
            )

    return styled_df
