from datetime import datetime, timedelta

import kagglehub
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from kagglehub import KaggleDatasetAdapter

st.set_page_config(layout="wide")

st.title("Fantasy Points Analyzer")
st.write("This is a tool to analyze fantasy points for a player over time.")



# Load data and player list
# result = load_and_process_data()
# if isinstance(result, tuple):
#     df_fp, player_list = result
# else:
#     # Fallback for backwards compatibility
#     df_fp = result
df_fp = pd.read_csv('PlayerStatistics_transformed_post_21.csv')
# player_list = sorted(df_fp["player_name"].unique()) if len(df_fp) > 0 else []

player_list = df_fp.groupby('player_name')['FP'].mean().sort_values(ascending=False).index.tolist()

# Check if data loaded successfully
if df_fp is None or len(df_fp) == 0:
    st.error("Failed to load dataset. Please try refreshing the page.")
    st.stop()

# display data
# st.dataframe(df_fp)

# display fantasy points over time
# st.line_chart(df_fp['FP'])

# display fantasy points over time for a specific player
# player_first_name = st.selectbox("Select a player", df_fp['firstName'].unique())
# player_last_name = st.selectbox("Select a player", df_fp['lastName'].unique())

# player_name = f"{player_first_name} {player_last_name}"
with st.sidebar:
    # Use cached player list to avoid repeated .unique() calls
    if not player_list:
        st.error("No players found in dataset.")
        st.stop()
    player_name = st.selectbox("Select a player", player_list)

    # Add date range filters
    col1, col2 = st.columns(2)
    with col1:
        start_date_filter = st.date_input(
            "Start Date",
            value=datetime(2024, 10, 22).date(),
            min_value=datetime(2015, 1, 1).date(),
            max_value=datetime.now().date(),
        )
    with col2:
        end_date_filter = st.date_input(
            "End Date",
            value=datetime.now().date(),
            min_value=datetime(2015, 1, 1).date(),
            max_value=datetime.now().date(),
        )

# Validate date range
if start_date_filter > end_date_filter:
    st.error("⚠️ Start date must be before or equal to end date. Please adjust your date selection.")
    st.stop()

# display fantasy points over time for a specific player
# Copy immediately to avoid SettingWithCopyWarning when modifying
try:
    player_data = df_fp[df_fp["player_name"] == player_name].copy()
except Exception as e:
    st.error(f"Error filtering player data: {e}")
    st.stop()

# Find and convert date column
date_col = None
for col in ["gameDateTimeEst", "game_date", "date", "GAME_DATE", "DATE_EST"]:
    if col in player_data.columns:
        date_col = col
        break

if date_col:
    if len(player_data) > 0:
        try:
            if not pd.api.types.is_datetime64_any_dtype(player_data[date_col]):
                player_data[date_col] = pd.to_datetime(player_data[date_col], errors="coerce", utc=True)
                if player_data[date_col].dt.tz is not None:
                    player_data[date_col] = player_data[date_col].dt.tz_localize(None)
            else:
                if player_data[date_col].dt.tz is not None:
                    player_data[date_col] = player_data[date_col].dt.tz_localize(None)

                nan_count = player_data[date_col].isna().sum()
                if nan_count > 0:
                    if "gameDateTimeEst_raw" in player_data.columns:
                        mask = player_data[date_col].isna()
                        if mask.sum() > 0:
                            player_data.loc[mask, date_col] = pd.to_datetime(
                                player_data.loc[mask, "gameDateTimeEst_raw"],
                                errors="coerce",
                                format="ISO8601",
                            )
                            if player_data[date_col].dt.tz is not None:
                                player_data[date_col] = player_data[date_col].dt.tz_localize(None)
        except Exception as e:
            st.warning(f"Warning: Could not process date column: {e}")

    # Drop NaN values
    try:
        player_data = player_data.dropna(subset=[date_col, "FP"])
    except Exception as e:
        st.error(f"Error dropping NaN values: {e}")
        st.stop()

    try:
        start_date = pd.Timestamp(start_date_filter).normalize()
        end_date = pd.Timestamp(end_date_filter).normalize()
        player_data["date_only"] = pd.to_datetime(player_data[date_col]).dt.normalize()

        player_data = player_data[
            (player_data["date_only"] >= start_date) & (player_data["date_only"] <= end_date)
        ]
        player_data = player_data.sort_values(date_col)
    except Exception as e:
        st.error(f"Error filtering by date range: {e}")
        st.stop()

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
                        window_data = player_data_ytd[player_data_ytd[date_col] >= cutoff_date].copy()

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

            game_log, stats = st.tabs(["Game Log", "Stats"])
            with game_log:
                st.subheader("Game Log")
                # Prepare dataframe for display - copy to avoid modification issues
                try:
                    display_df = player_data_ytd[
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
                    ].sort_values(by="game_loc_date", ascending=False).copy()

                    # Format numeric columns as integers (except FP, fieldGoalsPercentage, freeThrowsPercentage)
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
                                pd.to_numeric(display_df[col], errors="coerce").round().astype("Int64")
                            )

                    # Format decimal columns to 2 decimal places
                    decimal_columns = ["FP", "fieldGoalsPercentage", "freeThrowsPercentage"]
                    for col in decimal_columns:
                        if col in display_df.columns:
                            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)
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
                    "fieldGoalsPercentage",
                    "freeThrowsPercentage",
                ]

                # Only style columns that exist in the dataframe
                columns_to_style = [col for col in columns_to_style if col in display_df.columns]

                # Create styled dataframe with green gradient
                # Format decimal columns to always show 2 decimal places
                format_dict = {}
                for col in decimal_columns:
                    if col in display_df.columns:
                        format_dict[col] = "{:.2f}"

                styled_df = (
                    display_df.style.format(format_dict)
                    .background_gradient(
                        subset=columns_to_style,
                        cmap="Greens",
                        axis=0,  # Apply gradient along rows (column-wise)
                    )
                    .background_gradient(
                        subset="turnovers",
                        cmap="Reds",
                        axis=0,  # Apply gradient along rows (column-wise)
                    )
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
                        pd.isna(pm_values), "N/A",
                        np.where(pm_values > 0, "W",
                        np.where(pm_values < 0, "L", "T"))
                    )
                else:
                    win_loss = np.array(["N/A"] * len(player_data_ytd))

                # Create custom hover text - vectorized approach
                hover_texts = [
                    f"<b>Date:</b> {date}<br>"
                    f"<b>Opponent:</b> {opp}<br>"
                    f"<b>Result:</b> {wl}<br>"
                    f"<b>Fantasy Points:</b> {fp:.1f}"
                    for date, opp, wl, fp in zip(formatted_dates, opponent_info, win_loss, y)
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

            # Define moving average windows and colors
            ma_windows = {
                "7 days": ("7D", "#FF6B6B"),  # Red
                "14 days": ("14D", "#4ECDC4"),  # Teal
                "30 days": ("30D", "#45B7D1"),  # Blue
                "90 days": ("90D", "#FFA07A"),  # Light salmon
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
                if 'player_data_indexed' in locals():
                    del player_data_indexed

            # Create Plotly figure
            fig = go.Figure()

            # Add actual data line (green like Robinhood stock prices)
            scatter_kwargs = {
                "x": x,
                "y": y,
                "mode": "lines+markers+text",
                "name": "Fantasy Points",
                "line": {"color": "#00C805", "width": 0.5},  # Robinhood green
                "marker": {"color": "#00C805", "size": 4, "opacity": 0.6},
                "text": [f"{val:.1f}" for val in y],
                "textposition": "top center",
                "textfont": {"size": 9, "color": "#00C805"},
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

            # Update layout with dark background
            start_date_str = start_date_filter.strftime("%b %d, %Y")
            end_date_str = end_date_filter.strftime("%b %d, %Y")
            fig.update_layout(
                title={
                    "text": f"Fantasy Points Over Time: {player_name} ({start_date_str} - {end_date_str})",
                    "font": {"size": 20, "color": "white"},
                },
                xaxis={
                    "title": "Date",
                    "showgrid": True,
                    "gridcolor": "#333333",
                    "gridwidth": 1,
                    "showline": True,
                    "linecolor": "#555555",
                    "title_font": {"size": 14, "color": "#cccccc"},
                    "tickfont": {"color": "#cccccc"},
                },
                yaxis={
                    "title": "Fantasy Points",
                    "showgrid": True,
                    "gridcolor": "#333333",
                    "gridwidth": 1,
                    "showline": True,
                    "linecolor": "#555555",
                    "title_font": {"size": 14, "color": "#cccccc"},
                    "tickfont": {"color": "#cccccc"},
                },
                hovermode="closest",
                plot_bgcolor="black",
                paper_bgcolor="black",
                legend={
                    "yanchor": "top",
                    "y": 0.99,
                    "xanchor": "left",
                    "x": 0.01,
                    "bgcolor": "rgba(0, 0, 0, 0.8)",
                    "bordercolor": "#555555",
                    "borderwidth": 1,
                    "font": {"size": 12, "color": "white"},
                },
                margin={"l": 60, "r": 20, "t": 60, "b": 50},
            )

            st.plotly_chart(fig, width="stretch")

        else:
            st.write(
                f"No data available for {player_name} in the selected date range ({start_date_filter} to {end_date_filter})."
            )
    else:
        st.write(
            f"No data available for {player_name} in the selected date range ({start_date_filter} to {end_date_filter})."
        )
else:
    st.write("No date column found in the dataset. Cannot display time-based chart.")
