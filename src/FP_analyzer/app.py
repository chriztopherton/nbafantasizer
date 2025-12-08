from datetime import datetime, timedelta

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from injury_scraper import ESPNInjuryScraper

st.set_page_config(
    page_title="Fantasy Points Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for fantasy stat tracker look and feel
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

st.title("🏀 Fantasy Points Analyzer")
st.markdown("Analyze fantasy points for players over time and evaluate trades.")


# Initialize injury scraper (cached)
@st.cache_resource
def get_injury_scraper():
    return ESPNInjuryScraper()


injury_scraper = get_injury_scraper()

# Load data and player list
# result = load_and_process_data()
# if isinstance(result, tuple):
#     df_fp, player_list = result
# else:
#     # Fallback for backwards compatibility
#     df_fp = result
df_fp = pd.read_csv("PlayerStatistics_transformed_post_21.csv")
# player_list = sorted(df_fp["player_name"].unique()) if len(df_fp) > 0 else []

player_list = df_fp.groupby("player_name")["FP"].mean().sort_values(ascending=False).index.tolist()

# Check if data loaded successfully
if df_fp is None or len(df_fp) == 0:
    st.error("Failed to load dataset. Please try refreshing the page.")
    st.stop()


# Helper function to find and process date column
def find_and_process_date_column(data):
    """Find and process the date column in the dataframe"""
    date_col = None
    for col in ["gameDateTimeEst", "game_date", "date", "GAME_DATE", "DATE_EST"]:
        if col in data.columns:
            date_col = col
            break

    if date_col and len(data) > 0:
        try:
            if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                data[date_col] = pd.to_datetime(data[date_col], errors="coerce", utc=True)
                if data[date_col].dt.tz is not None:
                    data[date_col] = data[date_col].dt.tz_localize(None)
            else:
                if data[date_col].dt.tz is not None:
                    data[date_col] = data[date_col].dt.tz_localize(None)

            nan_count = data[date_col].isna().sum()
            if nan_count > 0:
                if "gameDateTimeEst_raw" in data.columns:
                    mask = data[date_col].isna()
                    if mask.sum() > 0:
                        data.loc[mask, date_col] = pd.to_datetime(
                            data.loc[mask, "gameDateTimeEst_raw"],
                            errors="coerce",
                            format="ISO8601",
                        )
                        if data[date_col].dt.tz is not None:
                            data[date_col] = data[date_col].dt.tz_localize(None)
        except Exception as e:
            st.warning(f"Warning: Could not process date column: {e}")

    return date_col


# Helper function to filter data by date range
def filter_data_by_date(data, date_col, start_date, end_date):
    """Filter dataframe by date range"""
    if date_col is None or len(data) == 0:
        return data

    try:
        data = data.dropna(subset=[date_col, "FP"])
        start_date_ts = pd.Timestamp(start_date).normalize()
        end_date_ts = pd.Timestamp(end_date).normalize()
        data["date_only"] = pd.to_datetime(data[date_col]).dt.normalize()
        data = data[(data["date_only"] >= start_date_ts) & (data["date_only"] <= end_date_ts)]
        data = data.sort_values(date_col)
    except Exception as e:
        st.warning(f"Warning: Could not filter by date range: {e}")

    return data


# Helper function to get player FP data for visualization
def get_player_fp_data(player_name, df_fp, date_col, start_date, end_date):
    """Get all FP values for a player within date range"""
    player_data = df_fp[df_fp["player_name"] == player_name].copy()

    if len(player_data) == 0:
        return None

    # Process date column
    date_col_processed = find_and_process_date_column(player_data)
    if date_col_processed is None:
        date_col_processed = date_col

    # Filter by date range
    player_data = filter_data_by_date(player_data, date_col_processed, start_date, end_date)

    if len(player_data) == 0:
        return None

    return player_data["FP"].values


# Helper function to calculate player statistics for trade analysis
def calculate_player_trade_stats(player_name, df_fp, date_col, start_date, end_date):
    """Calculate statistics for a player for trade analysis"""
    player_data = df_fp[df_fp["player_name"] == player_name].copy()

    if len(player_data) == 0:
        return None

    # Process date column
    date_col_processed = find_and_process_date_column(player_data)
    if date_col_processed is None:
        date_col_processed = date_col

    # Filter by date range
    player_data = filter_data_by_date(player_data, date_col_processed, start_date, end_date)

    if len(player_data) == 0:
        return None

    # Calculate statistics
    stats = {
        "player_name": player_name,
        "avg_fp": player_data["FP"].mean(),
        "std_fp": player_data["FP"].std(),
        "total_fp": player_data["FP"].sum(),
        "games_played": len(player_data),
        "min_fp": player_data["FP"].min(),
        "max_fp": player_data["FP"].max(),
        "median_fp": player_data["FP"].median(),
    }

    return stats


# display data
# st.dataframe(df_fp)

# display fantasy points over time
# st.line_chart(df_fp['FP'])

# display fantasy points over time for a specific player
# player_first_name = st.selectbox("Select a player", df_fp['firstName'].unique())
# player_last_name = st.selectbox("Select a player", df_fp['lastName'].unique())

# Create tabs for different features
tab1, tab2 = st.tabs(["Player Analysis", "Trade Analyzer"])

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)

    # Add button to refresh injury data cache
    if st.button("🔄 Refresh Injury Data"):
        injury_scraper.clear_cache()
        st.success("Injury data cache cleared! Data will refresh on next load.")

    st.markdown("---")

    # Player selection (shared across tabs)
    if not player_list:
        st.error("No players found in dataset.")
        st.stop()

    st.markdown("**Player Selection**")
    player_name = st.selectbox(
        "Select a player", player_list, key="player_select_sidebar", label_visibility="collapsed"
    )

    # Track the last selected player to detect changes
    if "last_selected_player" not in st.session_state:
        st.session_state.last_selected_player = player_name
        # Initialize team1_players with the selected player
        if player_name:
            st.session_state.team1_players = [player_name]

    # When player changes, add to team1_players if not already there
    if player_name != st.session_state.last_selected_player:
        st.session_state.last_selected_player = player_name
        if "team1_players" not in st.session_state:
            st.session_state.team1_players = [player_name] if player_name else []
        elif player_name and player_name not in st.session_state.team1_players:
            # Add the new player to the beginning of the list
            st.session_state.team1_players = [player_name] + st.session_state.team1_players

    # Add date range filters (shared across tabs)
    st.markdown("**Date Range**")
    col1, col2 = st.columns(2)
    with col1:
        start_date_filter = st.date_input(
            "Start Date",
            value=datetime(2024, 10, 22).date(),
            min_value=datetime(2015, 1, 1).date(),
            max_value=datetime.now().date(),
            label_visibility="collapsed",
        )
    with col2:
        end_date_filter = st.date_input(
            "End Date",
            value=datetime.now().date(),
            min_value=datetime(2015, 1, 1).date(),
            max_value=datetime.now().date(),
            label_visibility="collapsed",
        )

    # Validate date range
    if start_date_filter > end_date_filter:
        st.error("⚠️ Start date must be before or equal to end date.")
        st.stop()

# Tab 1: Player Analysis
with tab1:
    # display fantasy points over time for a specific player
    # Copy immediately to avoid SettingWithCopyWarning when modifying
    try:
        player_data = df_fp[df_fp["player_name"] == player_name].copy()
    except Exception as e:
        st.error(f"Error filtering player data: {e}")
        st.stop()

    # Find and convert date column
    date_col = find_and_process_date_column(player_data)

    if date_col:
        # Filter by date range
        player_data = filter_data_by_date(player_data, date_col, start_date_filter, end_date_filter)

        if len(player_data) > 0:
            # player_data is already filtered to the selected date range, so use it directly
            player_data_ytd = player_data

        if len(player_data_ytd) > 0:
            # Calculate aggregated averages for quantitative metrics over different time windows
            # Get the most recent date
            most_recent_date = player_data_ytd[date_col].max()

            # Define time windows (in days)
            time_windows = {
                "Last 7 days (avg)": 7,
                "Last 14 days (avg)": 14,
                "Last 30 days (avg)": 30,
                "Last 90 days (avg)": 90,
                "YTD (avg)": None,  # None means use all YTD data
            }

            # Build aggregated data for all time windows
            aggregated_rows = []
            try:
                for window_name, days in time_windows.items():
                    if days is None:
                        # YTD: use all data
                        window_data = player_data_ytd.copy()
                    else:
                        # Last N days: filter to most recent N days
                        cutoff_date = most_recent_date - timedelta(days=days)
                        window_data = player_data_ytd[
                            player_data_ytd[date_col] >= cutoff_date
                        ].copy()

                    if len(window_data) > 0:
                        row = {
                            "Time Window": window_name,
                            "Fantasy Points": window_data["FP"].mean(),
                            "Points": window_data["points"].mean(),
                            "Rebounds": window_data["reboundsTotal"].mean(),
                            "Assists": window_data["assists"].mean(),
                            "Blocks": window_data["blocks"].mean(),
                            "Steals": window_data["steals"].mean(),
                            "Turnovers": window_data["turnovers"].mean(),
                        }
                        aggregated_rows.append(row)
            except Exception as e:
                st.warning(f"Warning: Could not calculate all time window averages: {e}")

            # Create single DataFrame with all aggregated averages
            aggregated_df = pd.DataFrame(aggregated_rows)

            # Display injury report for selected player
            try:
                injury_info = injury_scraper.get_player_injury(player_name)
                if injury_info:
                    # st.info("🏥 **Injury Report**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Status", injury_info.get("status", "Unknown"))
                    with col2:
                        st.metric("Position", injury_info.get("position", "N/A"))
                    with col3:
                        return_date = injury_info.get("return_date", "")
                        if return_date:
                            st.metric("Expected Return", return_date)
                        else:
                            st.metric("Expected Return", "TBD")

                    # Get comment and description fields
                    comment = injury_info.get("comment", "")
                    description = injury_info.get("description", "")

                    # Debug: Show what we got (can be removed later)
                    # st.write(f"Debug - Comment: {comment[:50] if comment else 'None'}...")
                    # st.write(f"Debug - Description: {description[:50] if description else 'None'}...")

                    # Display comment field (primary source of detailed injury info)
                    if comment and comment.strip():
                        # st.markdown("---")
                        # st.markdown(f"**📝 Injury Comment:**")
                        st.info(comment)
                    # elif description and description.strip():
                    #     # Fallback to description if comment is not available
                    #     # st.markdown("---")
                    #     st.markdown(f"**📝 Injury Details:**")
                    #     st.info(description)
                    else:
                        # If neither comment nor description, show status only
                        st.markdown("---")
                        st.write(f"**Status:** {injury_info.get('status', 'Unknown')}")
                else:
                    st.success("✅ **Player Status:** Healthy - No current injuries reported")
            except Exception as e:
                st.warning(f"⚠️ Could not fetch injury information: {e}")
                import traceback

                st.error(f"Error details: {traceback.format_exc()}")

            game_log, stats = st.tabs(["Game Log", "Stats"])
            with game_log:
                st.subheader("Game Log")
                # Prepare dataframe for display - copy to avoid modification issues
                try:
                    display_df = (
                        player_data_ytd[
                            [
                                "game_loc_date",
                                "FP",
                                "numMinutes",
                                "points",
                                "reboundsTotal",
                                "assists",
                                "steals",
                                "blocks",
                                "turnovers",
                                "fieldGoalsPercentage",
                                "freeThrowsPercentage",
                                "plusMinusPoints",
                            ]
                        ]
                        .sort_values(by="game_loc_date", ascending=False)
                        .copy()
                    )

                    # Replace "Percentage" with "%" in column names
                    display_df.columns = [
                        col.replace("Percentage", "%") for col in display_df.columns
                    ]

                    # Format numeric columns as integers (except FP, fieldGoals%, freeThrows%)
                    integer_columns = [
                        "numMinutes",
                        "points",
                        "reboundsTotal",
                        "assists",
                        "steals",
                        "blocks",
                        "turnovers",
                        "plusMinusPoints",
                    ]
                    for col in integer_columns:
                        if col in display_df.columns:
                            # Round and convert to nullable integer type
                            display_df[col] = (
                                pd.to_numeric(display_df[col], errors="coerce")
                                .round()
                                .astype("Int64")
                            )

                    # Format decimal columns to 2 decimal places
                    decimal_columns = ["FP", "fieldGoals%", "freeThrows%"]
                    for col in decimal_columns:
                        if col in display_df.columns:
                            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(
                                2
                            )
                except Exception as e:
                    st.error(f"Error preparing display dataframe: {e}")
                    st.stop()

                # Apply green gradient styling to specified columns
                columns_to_style = [
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

                # Function to apply text color gradient instead of background
                def color_text_gradient(series, cmap_name="Greens"):
                    """Apply color gradient to text based on value in a Series"""
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

                # Create styled dataframe with text color gradient
                styled_df = display_df.style.format(format_dict)

                # Apply text color gradient to positive columns (darker greens for light theme)
                for col in columns_to_style:
                    if col in display_df.columns:
                        styled_df = styled_df.apply(
                            lambda x: color_text_gradient(x, cmap_name="Greens"),
                            subset=[col],
                            axis=0,
                        )

                # Apply red text color gradient to turnovers
                if "turnovers" in display_df.columns:
                    styled_df = styled_df.apply(
                        lambda x: color_text_gradient(x, cmap_name="Reds"),
                        subset=["turnovers"],
                        axis=0,
                    )

                st.dataframe(styled_df, width="stretch")
                # st.dataframe(player_data_ytd, use_container_width=True)
            with stats:
                st.subheader("Stats")
                # st.dataframe(player_data_ytd, use_container_width=True)

                # Display aggregated averages as a single table
                st.subheader("Aggregated Averages by Time Window")
                if len(aggregated_df) > 0:
                    st.dataframe(aggregated_df, width="stretch")

            # Use YTD data for the main chart
            x = player_data_ytd[date_col].values
            y = player_data_ytd["FP"].values

            # Prepare hover information: date, opponent, win/loss
            # Format dates for hover
            formatted_dates = pd.to_datetime(player_data_ytd[date_col]).dt.strftime("%b %d, %Y")

            # Get opponent information - use pre-computed column if available, otherwise compute
            try:
                if "opponent_with_at" in player_data_ytd.columns:
                    opponent_info = player_data_ytd["opponent_with_at"].values
                else:
                    if "home" in player_data_ytd.columns:
                        opponent_info = np.where(
                            player_data_ytd["home"] == 0,
                            "@" + player_data_ytd["opponentteamName"],
                            player_data_ytd["opponentteamName"],
                        )
                    else:
                        opponent_info = player_data_ytd["opponentteamName"].values

                # Determine win/loss from plusMinusPoints (positive = win, negative = loss) - vectorized
                if "plusMinusPoints" in player_data_ytd.columns:
                    pm_values = player_data_ytd["plusMinusPoints"].values
                    win_loss = np.where(
                        pd.isna(pm_values),
                        "N/A",
                        np.where(pm_values > 0, "W", np.where(pm_values < 0, "L", "T")),
                    )
                else:
                    win_loss = np.array(["N/A"] * len(player_data_ytd))

                # Create custom hover text - vectorized approach (styled for light theme)
                hover_texts = [
                    f"<b>Date:</b> {date}<br>"
                    f"<b>Opponent:</b> {opp}<br>"
                    f"<b>Result:</b> {wl}<br>"
                    f"<b>Fantasy Points:</b> {fp:.1f}"
                    for date, opp, wl, fp in zip(
                        formatted_dates, opponent_info, win_loss, y, strict=False
                    )
                ]
            except Exception as e:
                # Fallback to simple hover text if there's an error
                st.warning(f"Warning: Could not create custom hover text: {e}")
                hover_texts = None

            # Calculate moving averages for different time windows
            # Copy before setting index to avoid issues
            try:
                # Use a small copy for indexing to save memory
                player_data_indexed = player_data_ytd[[date_col, "FP"]].copy().set_index(date_col)
            except Exception as e:
                st.error(f"Error creating indexed dataframe: {e}")
                st.stop()

            # Define moving average windows and colors (optimized for light theme)
            ma_windows = {
                "7 days": ("7D", "#E74C3C"),  # Red
                "14 days": ("14D", "#3498DB"),  # Blue
                "30 days": ("30D", "#9B59B6"),  # Purple
                "90 days": ("90D", "#F39C12"),  # Orange
            }

            # Calculate moving averages
            moving_averages = {}
            try:
                for ma_name, (window_size, color) in ma_windows.items():
                    ma_series = (
                        player_data_indexed["FP"]
                        .rolling(window=window_size, center=True, min_periods=1)
                        .mean()
                    )
                    # Store only values and color to minimize memory
                    moving_averages[ma_name] = {"values": ma_series.values, "color": color}
                # Clean up the indexed dataframe after use
                del player_data_indexed
            except Exception as e:
                st.warning(f"Warning: Could not calculate all moving averages: {e}")
                # Clean up even on error
                if "player_data_indexed" in locals():
                    del player_data_indexed

            # Create Plotly figure
            fig = go.Figure()

            # Add actual data line (green for positive performance)
            scatter_kwargs = {
                "x": x,
                "y": y,
                "mode": "lines+markers+text",
                "name": "Fantasy Points",
                "line": {"color": "#27AE60", "width": 2},  # Green
                "marker": {"color": "#27AE60", "size": 5, "opacity": 0.7},
                "text": [f"{val:.1f}" for val in y],
                "textposition": "top center",
                "textfont": {"size": 9, "color": "#27AE60"},
            }
            # Only add custom hover if we successfully created it
            if hover_texts is not None:
                scatter_kwargs["hovertext"] = hover_texts
                scatter_kwargs["hovertemplate"] = "%{hovertext}<extra></extra>"

            fig.add_trace(go.Scatter(**scatter_kwargs))

            # Add moving average smoothing lines
            for ma_name, ma_data in moving_averages.items():
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=ma_data["values"],
                        mode="lines",
                        name=f"MA ({ma_name})",
                        line={"color": ma_data["color"], "width": 2},
                        opacity=0.8,
                    )
                )

            # Update layout with light background
            start_date_str = start_date_filter.strftime("%b %d, %Y")
            end_date_str = end_date_filter.strftime("%b %d, %Y")
            fig.update_layout(
                title={
                    "text": f"Fantasy Points Over Time: {player_name} ({start_date_str} - {end_date_str})",
                    "font": {"size": 20, "color": "#1A1A1A"},
                },
                xaxis={
                    "title": "Date",
                    "showgrid": True,
                    "gridcolor": "#E0E0E0",
                    "gridwidth": 1,
                    "showline": True,
                    "linecolor": "#CCCCCC",
                    "title_font": {"size": 14, "color": "#333333"},
                    "tickfont": {"color": "#666666"},
                },
                yaxis={
                    "title": "Fantasy Points",
                    "showgrid": True,
                    "gridcolor": "#E0E0E0",
                    "gridwidth": 1,
                    "showline": True,
                    "linecolor": "#CCCCCC",
                    "title_font": {"size": 14, "color": "#333333"},
                    "tickfont": {"color": "#666666"},
                },
                hovermode="closest",
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend={
                    "yanchor": "top",
                    "y": 0.99,
                    "xanchor": "left",
                    "x": 0.01,
                    "bgcolor": "rgba(255, 255, 255, 0.9)",
                    "bordercolor": "#E0E0E0",
                    "borderwidth": 1,
                    "font": {"size": 12, "color": "#1A1A1A"},
                },
                margin={"l": 60, "r": 20, "t": 60, "b": 50},
            )

            st.plotly_chart(fig, width="stretch")

        else:
            st.write(
                f"No data available for {player_name} in the selected date range ({start_date_filter} to {end_date_filter})."
            )
    else:
        st.write("No date column found in the dataset. Cannot display time-based chart.")

# Tab 2: Trade Analyzer
with tab2:
    st.header("Trade Analyzer")
    st.markdown(
        "Compare trade proposals between two teams. Analyze fantasy points, averages, and standard deviations to evaluate trade value."
    )

    # Find date column for trade analysis
    date_col_trade = find_and_process_date_column(df_fp.copy())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Team 1 - Players to Trade Away")
        # Session state is initialized in sidebar, so multiselect will use it automatically
        team1_players = st.multiselect(
            "Select players from Team 1", player_list, key="team1_players"
        )

    with col2:
        st.subheader("Team 2 - Players to Trade For")
        team2_players = st.multiselect(
            "Select players from Team 2", player_list, key="team2_players"
        )

    # Analyze trade button
    if st.button("Analyze Trade", type="primary", key="analyze_trade"):
        if not team1_players and not team2_players:
            st.warning("⚠️ Please select at least one player from either team.")
        else:
            # Calculate statistics for each player
            team1_stats = []
            team2_stats = []

            # Process Team 1 players
            for player in team1_players:
                stats = calculate_player_trade_stats(
                    player, df_fp, date_col_trade, start_date_filter, end_date_filter
                )
                if stats:
                    team1_stats.append(stats)
                else:
                    st.warning(f"⚠️ No data found for {player} in the selected date range.")

            # Process Team 2 players
            for player in team2_players:
                stats = calculate_player_trade_stats(
                    player, df_fp, date_col_trade, start_date_filter, end_date_filter
                )
                if stats:
                    team2_stats.append(stats)
                else:
                    st.warning(f"⚠️ No data found for {player} in the selected date range.")

            if team1_stats or team2_stats:
                # Create comparison report
                st.markdown("---")
                st.subheader("Trade Analysis Report")

                # Calculate totals
                team1_total_avg = sum(s["avg_fp"] for s in team1_stats) if team1_stats else 0
                team2_total_avg = sum(s["avg_fp"] for s in team2_stats) if team2_stats else 0

                team1_total_std = (
                    np.sqrt(sum(s["std_fp"] ** 2 for s in team1_stats)) if team1_stats else 0
                )
                team2_total_std = (
                    np.sqrt(sum(s["std_fp"] ** 2 for s in team2_stats)) if team2_stats else 0
                )

                team1_total_fp = sum(s["total_fp"] for s in team1_stats) if team1_stats else 0
                team2_total_fp = sum(s["total_fp"] for s in team2_stats) if team2_stats else 0

                # Display summary metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Team 1 Avg FP", f"{team1_total_avg:.2f}")
                    st.metric("Team 1 Total FP", f"{team1_total_fp:.2f}")
                with metric_col2:
                    st.metric("Team 2 Avg FP", f"{team2_total_avg:.2f}")
                    st.metric("Team 2 Total FP", f"{team2_total_fp:.2f}")
                with metric_col3:
                    diff = team1_total_avg - team2_total_avg
                    st.metric("Difference (Team 1 - Team 2)", f"{diff:.2f}", delta=f"{diff:.2f}")

                # Detailed player breakdown
                st.markdown("---")
                st.subheader("Player Breakdown")

                breakdown_col1, breakdown_col2 = st.columns(2)

                with breakdown_col1:
                    st.write("**Team 1 Players:**")
                    if team1_stats:
                        team1_df = pd.DataFrame(team1_stats)
                        team1_df = team1_df.round(2)
                        st.dataframe(
                            team1_df[
                                ["player_name", "avg_fp", "std_fp", "total_fp", "games_played"]
                            ],
                            use_container_width=True,
                        )
                    else:
                        st.write("No players selected")

                with breakdown_col2:
                    st.write("**Team 2 Players:**")
                    if team2_stats:
                        team2_df = pd.DataFrame(team2_stats)
                        team2_df = team2_df.round(2)
                        st.dataframe(
                            team2_df[
                                ["player_name", "avg_fp", "std_fp", "total_fp", "games_played"]
                            ],
                            use_container_width=True,
                        )
                    else:
                        st.write("No players selected")

                # Box plot visualization
                st.markdown("---")
                st.subheader("Fantasy Points Distribution (Box Plot)")
                st.write(
                    "Compare the distribution of fantasy points for each player. Box plots show quartiles, median, mean, and outliers."
                )

                # Collect FP data for all players
                team1_fp_data = {}
                team2_fp_data = {}

                for player in team1_players:
                    fp_values = get_player_fp_data(
                        player, df_fp, date_col_trade, start_date_filter, end_date_filter
                    )
                    if fp_values is not None and len(fp_values) > 0:
                        team1_fp_data[player] = fp_values

                for player in team2_players:
                    fp_values = get_player_fp_data(
                        player, df_fp, date_col_trade, start_date_filter, end_date_filter
                    )
                    if fp_values is not None and len(fp_values) > 0:
                        team2_fp_data[player] = fp_values

                # Create box plot if we have data
                if team1_fp_data or team2_fp_data:
                    fig_box = go.Figure()

                    # Colors for teams (optimized for light theme)
                    team1_color = "#E74C3C"  # Red
                    team2_color = "#3498DB"  # Blue

                    # Add Team 1 players
                    if team1_fp_data:
                        for player, fp_values in team1_fp_data.items():
                            # Calculate statistics for annotation
                            mean_fp = np.mean(fp_values)
                            median_fp = np.median(fp_values)

                            fig_box.add_trace(
                                go.Box(
                                    y=fp_values,
                                    name=player,
                                    boxmean="sd",  # Show mean and standard deviation
                                    boxpoints="outliers",  # Show outliers
                                    marker_color=team1_color,
                                    marker_opacity=0.7,
                                    line_color=team1_color,
                                    fillcolor="rgba(255, 107, 107, 0.3)",
                                    hovertemplate=f"<b>{player}</b> (Team 1)<br>"
                                    + "Q1: %{{q1:.2f}}<br>"
                                    + "Median: %{{median:.2f}}<br>"
                                    + "Q3: %{{q3:.2f}}<br>"
                                    + "Mean: "
                                    + f"{mean_fp:.2f}<br>"
                                    + "Min: %{{lowerfence:.2f}}<br>"
                                    + "Max: %{{upperfence:.2f}}<br>"
                                    + "<extra></extra>",
                                )
                            )

                    # Add Team 2 players
                    if team2_fp_data:
                        for player, fp_values in team2_fp_data.items():
                            # Calculate statistics for annotation
                            mean_fp = np.mean(fp_values)
                            median_fp = np.median(fp_values)

                            fig_box.add_trace(
                                go.Box(
                                    y=fp_values,
                                    name=player,
                                    boxmean="sd",  # Show mean and standard deviation
                                    boxpoints="outliers",  # Show outliers
                                    marker_color=team2_color,
                                    marker_opacity=0.7,
                                    line_color=team2_color,
                                    fillcolor="rgba(78, 205, 196, 0.3)",
                                    hovertemplate=f"<b>{player}</b> (Team 2)<br>"
                                    + "Q1: %{{q1:.2f}}<br>"
                                    + "Median: %{{median:.2f}}<br>"
                                    + "Q3: %{{q3:.2f}}<br>"
                                    + "Mean: "
                                    + f"{mean_fp:.2f}<br>"
                                    + "Min: %{{lowerfence:.2f}}<br>"
                                    + "Max: %{{upperfence:.2f}}<br>"
                                    + "<extra></extra>",
                                )
                            )

                    # Update layout with light background
                    fig_box.update_layout(
                        title={
                            "text": "Fantasy Points Distribution by Player",
                            "font": {"size": 18, "color": "#1A1A1A"},
                        },
                        xaxis={
                            "title": "Player",
                            "showgrid": True,
                            "gridcolor": "#E0E0E0",
                            "title_font": {"size": 14, "color": "#333333"},
                            "tickfont": {"color": "#666666"},
                        },
                        yaxis={
                            "title": "Fantasy Points",
                            "showgrid": True,
                            "gridcolor": "#E0E0E0",
                            "title_font": {"size": 14, "color": "#333333"},
                            "tickfont": {"color": "#666666"},
                        },
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        legend={
                            "yanchor": "top",
                            "y": 0.99,
                            "xanchor": "left",
                            "x": 0.01,
                            "bgcolor": "rgba(255, 255, 255, 0.9)",
                            "bordercolor": "#E0E0E0",
                            "borderwidth": 1,
                            "font": {"size": 11, "color": "#1A1A1A"},
                        },
                        margin={"l": 60, "r": 20, "t": 60, "b": 100},
                        height=500,
                    )

                    # Rotate x-axis labels for better readability
                    fig_box.update_xaxes(tickangle=-45)

                    st.plotly_chart(fig_box, use_container_width=True)

                    # Add explanation
                    st.caption(
                        "📊 **Box Plot Guide:** The box shows quartiles (Q1, median, Q3). The whiskers extend to min/max values within 1.5×IQR. Points outside are outliers. The mean is shown as a dashed line."
                    )
                else:
                    st.info(
                        "No fantasy points data available for the selected players in the chosen date range."
                    )

                # Trade rating
                st.markdown("---")
                st.subheader("Trade Rating")

                # Calculate rating based on multiple factors
                rating_score = 0
                rating_factors = []

                # Factor 1: Average FP difference (weighted heavily)
                if team1_total_avg > 0 and team2_total_avg > 0:
                    fp_diff_pct = (
                        (team1_total_avg - team2_total_avg)
                        / max(team1_total_avg, team2_total_avg)
                        * 100
                    )
                    if abs(fp_diff_pct) < 5:  # Within 5% is fair
                        rating_score += 3
                        rating_factors.append("✅ Average FP values are very close (fair trade)")
                    elif abs(fp_diff_pct) < 10:  # Within 10% is reasonable
                        rating_score += 2
                        rating_factors.append(
                            "⚠️ Average FP values differ by ~10% (slight imbalance)"
                        )
                    elif abs(fp_diff_pct) < 20:  # Within 20% is questionable
                        rating_score += 1
                        rating_factors.append("⚠️ Average FP values differ significantly (~20%)")
                    else:  # More than 20% difference
                        rating_score += 0
                        rating_factors.append("❌ Large difference in average FP values (>20%)")

                # Factor 2: Consistency (lower std dev is better)
                if team1_total_std > 0 and team2_total_std > 0:
                    std_ratio = min(team1_total_std, team2_total_std) / max(
                        team1_total_std, team2_total_std
                    )
                    if std_ratio > 0.8:
                        rating_score += 1
                        rating_factors.append("✅ Both sides have similar consistency")
                    elif std_ratio > 0.6:
                        rating_score += 0.5
                        rating_factors.append("⚠️ Some difference in consistency")
                    else:
                        rating_factors.append("⚠️ Significant difference in consistency")

                # Factor 3: Sample size (more games = more reliable)
                team1_games = sum(s["games_played"] for s in team1_stats) if team1_stats else 0
                team2_games = sum(s["games_played"] for s in team2_stats) if team2_stats else 0
                if team1_games >= 10 and team2_games >= 10:
                    rating_score += 1
                    rating_factors.append("✅ Sufficient sample size for both sides")
                elif team1_games >= 5 and team2_games >= 5:
                    rating_score += 0.5
                    rating_factors.append("⚠️ Limited sample size - results may be less reliable")
                else:
                    rating_factors.append("⚠️ Very limited sample size - use caution")

                # Determine final rating
                max_score = 5
                rating_pct = (rating_score / max_score) * 100

                if rating_pct >= 80:
                    rating = "✅ **WORTHY OF CONSIDERATION**"
                    rating_color = "success"
                elif rating_pct >= 60:
                    rating = "⚠️ **MARGINALLY WORTHY**"
                    rating_color = "warning"
                else:
                    rating = "❌ **NOT WORTHY OF CONSIDERATION**"
                    rating_color = "error"

                # Display rating
                if rating_color == "success":
                    st.success(rating)
                elif rating_color == "warning":
                    st.warning(rating)
                else:
                    st.error(rating)

                # Display rating factors
                st.write("**Rating Factors:**")
                for factor in rating_factors:
                    st.write(factor)

                # Additional insights
                st.markdown("---")
                st.subheader("Additional Insights")

                if team1_stats and team2_stats:
                    # Best and worst players
                    team1_best = max(team1_stats, key=lambda x: x["avg_fp"])
                    team2_best = max(team2_stats, key=lambda x: x["avg_fp"])

                    st.write("**Highest Average FP:**")
                    st.write(
                        f"- Team 1: {team1_best['player_name']} ({team1_best['avg_fp']:.2f} FP/game)"
                    )
                    st.write(
                        f"- Team 2: {team2_best['player_name']} ({team2_best['avg_fp']:.2f} FP/game)"
                    )

                    # Consistency comparison
                    team1_avg_std = np.mean([s["std_fp"] for s in team1_stats])
                    team2_avg_std = np.mean([s["std_fp"] for s in team2_stats])

                    st.write("**Consistency (Avg Std Dev):**")
                    st.write(f"- Team 1: {team1_avg_std:.2f}")
                    st.write(f"- Team 2: {team2_avg_std:.2f}")

                    if team1_avg_std < team2_avg_std:
                        st.info("Team 1 players are more consistent (lower variance)")
                    elif team2_avg_std < team1_avg_std:
                        st.info("Team 2 players are more consistent (lower variance)")
                    else:
                        st.info("Both teams have similar consistency")

                # Recommendation
                st.markdown("---")
                st.subheader("Recommendation")

                if diff > 5:
                    st.info(
                        f"**Team 1 is giving up more value** (avg {diff:.2f} FP/game more). Consider asking for additional compensation or reconsider the trade."
                    )
                elif diff < -5:
                    st.info(
                        f"**Team 2 is giving up more value** (avg {abs(diff):.2f} FP/game more). This trade favors Team 1."
                    )
                else:
                    st.success(
                        "**The trade appears relatively balanced** based on average fantasy points. Consider other factors like roster needs, positional depth, and future schedules."
                    )
