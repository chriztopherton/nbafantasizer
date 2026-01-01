"""Player Analysis tab functionality for Fantasy Points Analyzer."""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ai_analysis import render_ai_analysis_tab
from nba_stats import get_player_live_stats, render_nba_stats_section
from player_profile import render_player_profile_header
from style import apply_game_log_styling
from utils import (
    analyze_recent_games,
    filter_data_by_date,
    find_and_process_date_column,
    get_player_person_id,
)


def render_player_analysis_tab(
    player_name, df_fp, injury_scraper, start_date_filter, end_date_filter
):
    """
    Render the Player Analysis tab with injury info, game log, AI analysis, and chart.

    Args:
        player_name (str): Name of the selected player.
        df_fp (pd.DataFrame): DataFrame containing all player data.
        injury_scraper: Injury scraper instance for fetching injury information.
        start_date_filter: Start date for filtering data.
        end_date_filter: End date for filtering data.
    """
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

            # Get player ID for profile header and live stats
            player_id = get_player_person_id(player_name, df_fp)

            # Display player profile header
            render_player_profile_header(player_name, df_fp, player_id)

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

                    # Add second row for team and updated date if available
                    team = injury_info.get("team", "")
                    updated = injury_info.get("updated", "")
                    if team or updated:
                        col4, col5 = st.columns(2)
                        with col4:
                            if team:
                                st.metric("Team", team)
                        with col5:
                            if updated:
                                st.metric("Last Updated", updated)

                    # Get comment and description fields
                    comment = injury_info.get("comment", "")
                    description = injury_info.get("description", "")

                    # Get recent games performance analysis
                    recent_analysis = None
                    try:
                        recent_analysis = analyze_recent_games(
                            player_data_ytd, date_col, num_games=7
                        )
                    except Exception:
                        # Silently fail if analysis can't be generated
                        pass

                    # Combine injury comment with performance analysis for better flow
                    combined_info_parts = []

                    # Add injury comment/description first
                    if comment and comment.strip():
                        combined_info_parts.append(f"**Injury Status:** {comment}")
                    elif description and description.strip():
                        combined_info_parts.append(f"**Injury Details:** {description}")

                    # Add performance analysis if available
                    if recent_analysis:
                        if combined_info_parts:
                            combined_info_parts.append("")  # Add spacing
                            combined_info_parts.append("---")
                            combined_info_parts.append("")
                        combined_info_parts.append(recent_analysis)

                    # Display combined information
                    if combined_info_parts:
                        combined_text = "\n".join(combined_info_parts)
                        st.info(combined_text)
                    elif comment or description:
                        # Fallback: just show comment/description if no analysis
                        st.info(comment if comment and comment.strip() else description)
                    else:
                        # If neither comment nor description, show status only
                        st.markdown("---")
                        st.write(f"**Status:** {injury_info.get('status', 'Unknown')}")
                        # Still show performance analysis if available
                        if recent_analysis:
                            st.markdown("---")
                            st.info(recent_analysis)
                else:
                    # Player is healthy - show status with integrated performance analysis
                    try:
                        recent_analysis = analyze_recent_games(
                            player_data_ytd, date_col, num_games=7
                        )
                        if recent_analysis:
                            # Combine healthy status with performance analysis
                            combined_text = f"✅ **Player Status:** Healthy - No current injuries reported\n\n---\n\n{recent_analysis}"
                            st.info(combined_text)
                        else:
                            st.success(
                                "✅ **Player Status:** Healthy - No current injuries reported"
                            )
                    except Exception:
                        # Fallback to just status if analysis fails
                        st.success("✅ **Player Status:** Healthy - No current injuries reported")
            except Exception as e:
                st.warning(f"⚠️ Could not fetch injury information: {e}")
                import traceback

                st.error(f"Error details: {traceback.format_exc()}")

                # Still show performance analysis even if injury fetch fails
                try:
                    recent_analysis = analyze_recent_games(player_data_ytd, date_col, num_games=7)
                    if recent_analysis:
                        st.markdown("---")
                        st.markdown("**📊 Recent Performance Analysis:**")
                        st.info(recent_analysis)
                except Exception:
                    # Silently fail if analysis can't be generated
                    pass

            # Get player ID for NBA stats
            player_id = get_player_person_id(player_name, df_fp)

            game_log, season_stats, ai_analysis = st.tabs(
                ["Game Log", "Season Stats", "🤖 AI Analysis (beta)"]
            )
            with game_log:
                _render_game_log_tab(player_data_ytd, aggregated_df, player_id)

            with season_stats:
                render_nba_stats_section(player_id, player_name)

            with ai_analysis:
                render_ai_analysis_tab(player_name, player_data_ytd, aggregated_df, injury_scraper)

            # Use YTD data for the main chart
            _render_fantasy_points_chart(
                player_data_ytd, date_col, player_name, start_date_filter, end_date_filter
            )

        else:
            st.write(
                f"No data available for {player_name} in the selected date range ({start_date_filter} to {end_date_filter})."
            )
    else:
        st.write("No date column found in the dataset. Cannot display time-based chart.")


def _render_game_log_tab(player_data_ytd, aggregated_df, player_id=None):
    """
    Render the Game Log sub-tab with player statistics, including live stats as top row.

    Args:
        player_data_ytd (pd.DataFrame): Filtered player data for the date range.
        aggregated_df (pd.DataFrame): Aggregated statistics by time window.
        player_id (int, optional): NBA player ID for live stats.
    """
    st.subheader("2025 Season Game Log")

    # Check for live stats and add as first row if player is playing
    live_stats_row = None
    if player_id:
        live_stats = get_player_live_stats(player_id)
        if live_stats:
            live_stats_row = _create_live_stats_row(live_stats)

    # Prepare dataframe for display - copy to avoid modification issues
    try:
        # Get all needed columns
        base_df = player_data_ytd.copy()

        # Parse date and opponent from game_loc_date or separate columns
        if "opponent_with_at" in base_df.columns:
            base_df["Opp"] = base_df["opponent_with_at"]
        elif "opponentteamName" in base_df.columns and "home" in base_df.columns:
            base_df["Opp"] = base_df.apply(
                lambda row: (
                    f"@{row['opponentteamName']}"
                    if row.get("home") == 0
                    else row["opponentteamName"]
                ),
                axis=1,
            )
        elif "opponentteamName" in base_df.columns:
            base_df["Opp"] = base_df["opponentteamName"]
        else:
            base_df["Opp"] = "N/A"

        # Format date column
        date_col = find_and_process_date_column(base_df)
        if date_col:
            base_df["Date"] = pd.to_datetime(base_df[date_col]).dt.strftime("%b %d, %Y")
        else:
            base_df["Date"] = "N/A"

        # Create Status column (game outcome or scheduled time)
        base_df["Status"] = "-"
        if "plusMinusPoints" in base_df.columns:
            # For completed games, show outcome
            def get_status(row):
                if pd.notna(row.get("plusMinusPoints")):
                    # This is a completed game - we'd need score info, but for now show W/L
                    pm = row["plusMinusPoints"]
                    if pm > 0:
                        return "W"
                    elif pm < 0:
                        return "L"
                    else:
                        return "T"
                return "-"

            base_df["Status"] = base_df.apply(get_status, axis=1)

        # Format minutes as MM:SS
        if "numMinutes" in base_df.columns:

            def format_minutes(min_val):
                if pd.isna(min_val) or min_val == 0:
                    return "-"
                try:
                    # Convert to float if string
                    min_float = float(min_val)
                    minutes = int(min_float)
                    seconds = int((min_float - minutes) * 60)
                    return f"{minutes}:{seconds:02d}"
                except (ValueError, TypeError):
                    return "-"

            base_df["MIN"] = base_df["numMinutes"].apply(format_minutes)
        else:
            base_df["MIN"] = "-"

        # Rename columns for display
        column_mapping = {
            "FP": "Fan Pts",
            "points": "PTS",
            "reboundsTotal": "REB",
            "assists": "AST",
            "steals": "ST",
            "blocks": "BLK",
            "turnovers": "TO",
        }

        # Select and rename columns in the desired order
        display_columns = [
            "Date",
            "Opp",
            "Status",
            "Fan Pts",
            "MIN",
            "PTS",
            "REB",
            "AST",
            "ST",
            "BLK",
            "TO",
        ]

        # Build the display dataframe
        display_df = pd.DataFrame()
        for col in display_columns:
            if col in ["Date", "Opp", "Status", "MIN"]:
                # These are already created above
                if col in base_df.columns:
                    display_df[col] = base_df[col]
                else:
                    display_df[col] = "-"
            elif col == "Fan Pts":
                display_df[col] = base_df["FP"] if "FP" in base_df.columns else 0
            else:
                # Map column names
                original_col = None
                for orig, mapped in column_mapping.items():
                    if mapped == col:
                        original_col = orig
                        break
                if original_col and original_col in base_df.columns:
                    display_df[col] = base_df[original_col]
                else:
                    display_df[col] = 0

        # Sort by date descending (most recent first)
        if "Date" in display_df.columns and date_col:
            # Create a temporary sort column with actual datetime
            display_df["_sort_date"] = pd.to_datetime(base_df[date_col], errors="coerce")
            display_df = display_df.sort_values("_sort_date", ascending=False, na_position="last")
            display_df = display_df.drop("_sort_date", axis=1)

        # Format numeric columns
        integer_columns = ["PTS", "REB", "AST", "ST", "BLK", "TO"]
        for col in integer_columns:
            if col in display_df.columns:
                display_df[col] = (
                    pd.to_numeric(display_df[col], errors="coerce").fillna(0).astype(int)
                )

        # Format Fan Pts to 2 decimal places
        if "Fan Pts" in display_df.columns:
            display_df["Fan Pts"] = pd.to_numeric(display_df["Fan Pts"], errors="coerce").round(2)

        # Add live stats row at the top if available
        has_live_stats = live_stats_row is not None
        if has_live_stats:
            # Prepend live stats row to the dataframe
            display_df = pd.concat([live_stats_row, display_df], ignore_index=True)
    except Exception as e:
        st.error(f"Error preparing display dataframe: {e}")
        st.stop()

    # Apply styling to game log dataframe
    styled_df = apply_game_log_styling(display_df, highlight_live_row=has_live_stats)
    st.dataframe(styled_df, width="stretch", hide_index=True)

    # Display aggregated averages as a single table
    st.subheader("Averages")
    if len(aggregated_df) > 0:
        st.dataframe(aggregated_df, width="stretch")


def _create_live_stats_row(live_stats):
    """
    Create a DataFrame row for live stats that matches the game log format.

    Args:
        live_stats (dict): Live stats dictionary from get_player_live_stats.

    Returns:
        pd.DataFrame: Single-row DataFrame with live stats formatted for game log.
    """
    player_data = live_stats.get("player", {})
    if not player_data:
        return None

    # Extract stats from player data
    stats = player_data.get("statistics", {})
    if not stats:
        stats = player_data

    # Extract key stats
    pts = stats.get("points") or stats.get("pts") or stats.get("PTS") or stats.get("score") or 0
    reb = stats.get("rebounds") or stats.get("reb") or stats.get("REB") or 0
    ast = stats.get("assists") or stats.get("ast") or stats.get("AST") or 0
    stl = stats.get("steals") or stats.get("stl") or stats.get("STL") or 0
    blk = stats.get("blocks") or stats.get("blk") or stats.get("BLK") or 0
    tov = stats.get("turnovers") or stats.get("tov") or stats.get("TOV") or 0
    min_played = (
        stats.get("minutes")
        or stats.get("min")
        or stats.get("MIN")
        or stats.get("minutesCalculated")
        or "0:00"
    )

    # Calculate fantasy points (same formula as in export_post_21_data.py)
    fp = pts * 1.0 + reb * 1.2 + ast * 1.5 + blk * 3.0 + stl * 3.0 + tov * -1.0

    # Get game info
    opponent = live_stats.get("opponent", "Unknown")
    game_status = live_stats.get("gameStatus", "LIVE")

    # Format minutes (convert string like "30:41" to float if needed)
    if isinstance(min_played, str):
        try:
            # Parse "MM:SS" format
            parts = min_played.split(":")
            if len(parts) == 2:
                min_played = float(parts[0]) + float(parts[1]) / 60
            else:
                min_played = float(min_played)
        except (ValueError, AttributeError):
            min_played = 0.0

    # Format minutes as MM:SS
    if isinstance(min_played, str):
        # Already in MM:SS format
        min_formatted = min_played if ":" in min_played else "0:00"
    else:
        # Convert float to MM:SS
        try:
            min_float = float(min_played)
            minutes = int(min_float)
            seconds = int((min_float - minutes) * 60)
            min_formatted = f"{minutes}:{seconds:02d}"
        except (ValueError, TypeError):
            min_formatted = "0:00"

    # Create date string
    from datetime import datetime

    now = datetime.now()
    live_date = now.strftime("%b %d, %Y")

    # Create row matching new game log format
    live_row = pd.DataFrame(
        {
            "Date": [live_date],
            "Opp": [opponent],
            "Status": [f"🔴 {game_status}"],
            "Fan Pts": [round(fp, 2)],
            "MIN": [min_formatted],
            "PTS": [int(pts)],
            "REB": [int(reb)],
            "AST": [int(ast)],
            "ST": [int(stl)],
            "BLK": [int(blk)],
            "TO": [int(tov)],
        }
    )

    return live_row


def _render_fantasy_points_chart(
    player_data_ytd, date_col, player_name, start_date_filter, end_date_filter
):
    """
    Render the fantasy points over time chart with moving averages.

    Args:
        player_data_ytd (pd.DataFrame): Filtered player data for the date range.
        date_col (str): Name of the date column.
        player_name (str): Name of the player.
        start_date_filter: Start date for the chart title.
        end_date_filter: End date for the chart title.
    """
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
            for date, opp, wl, fp in zip(formatted_dates, opponent_info, win_loss, y, strict=False)
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
            try:
                del player_data_indexed
            except NameError:
                pass

    # Add radio button to select moving average
    ma_options = ["None"] + list(ma_windows.keys())
    selected_ma = st.radio(
        "Select Moving Average Trend Line:",
        options=ma_options,
        index=1,
        horizontal=True,
        key="ma_selector",
    )

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

    # Add selected moving average smoothing line (if any)
    if selected_ma != "None" and selected_ma in moving_averages:
        ma_data = moving_averages[selected_ma]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ma_data["values"],
                mode="lines",
                name=f"MA ({selected_ma})",
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
