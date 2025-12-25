"""Trade Analyzer tab functionality for Fantasy Points Analyzer."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ai_model import LANGCHAIN_AVAILABLE, PlayerAnalysisChatbot
from utils import calculate_player_trade_stats, find_and_process_date_column, get_player_fp_data


def render_trade_analyzer_tab(
    df_fp, player_list, injury_scraper, start_date_filter, end_date_filter
):
    """
    Render the Trade Analyzer tab with player selection, analysis, and AI insights.

    Args:
        df_fp (pd.DataFrame): DataFrame containing all player data.
        player_list (list): List of available player names.
        injury_scraper: Injury scraper instance for fetching injury information.
        start_date_filter: Start date for filtering data.
        end_date_filter: End date for filtering data.
    """
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

                # Store all calculated values in session state for persistence
                st.session_state.current_team1_stats = team1_stats
                st.session_state.current_team2_stats = team2_stats
                st.session_state.current_team1_players = team1_players
                st.session_state.current_team2_players = team2_players
                st.session_state.current_team1_total_avg = team1_total_avg
                st.session_state.current_team2_total_avg = team2_total_avg
                st.session_state.current_team1_total_std = team1_total_std
                st.session_state.current_team2_total_std = team2_total_std
                st.session_state.current_team1_total_fp = team1_total_fp
                st.session_state.current_team2_total_fp = team2_total_fp
                st.session_state.trade_analyzed = True

                # Clear old AI analysis when new trade is analyzed
                if "trade_analysis" in st.session_state:
                    del st.session_state.trade_analysis
                if "trade_analysis_generated" in st.session_state:
                    del st.session_state.trade_analysis_generated
                if "trade_chat_messages" in st.session_state:
                    del st.session_state.trade_chat_messages

    # Display trade analysis if it exists in session state
    if "trade_analyzed" in st.session_state and st.session_state.trade_analyzed:
        # Retrieve from session state
        team1_stats = st.session_state.get("current_team1_stats", [])
        team2_stats = st.session_state.get("current_team2_stats", [])
        team1_players = st.session_state.get("current_team1_players", [])
        team2_players = st.session_state.get("current_team2_players", [])
        team1_total_avg = st.session_state.get("current_team1_total_avg", 0)
        team2_total_avg = st.session_state.get("current_team2_total_avg", 0)
        team1_total_std = st.session_state.get("current_team1_total_std", 0)
        team2_total_std = st.session_state.get("current_team2_total_std", 0)
        team1_total_fp = st.session_state.get("current_team1_total_fp", 0)
        team2_total_fp = st.session_state.get("current_team2_total_fp", 0)

        if team1_stats or team2_stats:
            # Create comparison report
            st.markdown("---")
            st.subheader("Trade Analysis Report")

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
            _render_player_breakdown(team1_stats, team2_stats, injury_scraper)

            # Box plot visualization
            _render_box_plot(
                team1_players,
                team2_players,
                df_fp,
                date_col_trade,
                start_date_filter,
                end_date_filter,
            )

            # Trade rating
            _render_trade_rating(
                team1_stats,
                team2_stats,
                team1_total_avg,
                team2_total_avg,
                team1_total_std,
                team2_total_std,
            )

            # Additional insights
            _render_additional_insights(team1_stats, team2_stats, team1_total_avg, team2_total_avg)

            # AI Trade Analysis
            _render_ai_trade_analysis(
                team1_stats,
                team2_stats,
                team1_players,
                team2_players,
                team1_total_avg,
                team2_total_avg,
                team1_total_std,
                team2_total_std,
                injury_scraper,
            )


def _render_player_breakdown(team1_stats, team2_stats, injury_scraper):
    """Render the player breakdown section with health status."""
    st.markdown("---")
    st.subheader("Player Breakdown")

    breakdown_col1, breakdown_col2 = st.columns(2)

    with breakdown_col1:
        st.write("**Team 1 Players:**")
        if team1_stats:
            _render_team_stats(team1_stats, injury_scraper)
        else:
            st.write("No players selected")

    with breakdown_col2:
        st.write("**Team 2 Players:**")
        if team2_stats:
            _render_team_stats(team2_stats, injury_scraper)
        else:
            st.write("No players selected")


def _render_team_stats(team_stats, injury_scraper):
    """Render statistics table for a team with injury information."""
    # Fetch injury info for each player
    for stats in team_stats:
        try:
            injury_info = injury_scraper.get_player_injury(stats["player_name"])
            if injury_info:
                stats["health_status"] = injury_info.get("status", "Unknown")
                injury_comment = injury_info.get("comment", "")
                if injury_comment:
                    # Truncate long comments
                    stats["health_note"] = (
                        injury_comment[:100] + "..."
                        if len(injury_comment) > 100
                        else injury_comment
                    )
                else:
                    stats["health_note"] = stats["health_status"]
            else:
                stats["health_status"] = "Healthy"
                stats["health_note"] = "No current injuries reported"
        except Exception:
            stats["health_status"] = "Unknown"
            stats["health_note"] = "Could not fetch injury data"

    team_df = pd.DataFrame(team_stats)
    team_df = team_df.round(2)

    # Select and order columns for display
    display_columns = [
        "player_name",
        "health_status",
        "health_note",
        "games_played",
        "avg_fp",
        "avg_points",
        "avg_rebounds",
        "avg_assists",
        "avg_steals",
        "avg_blocks",
        "avg_turnovers",
        "avg_minutes",
        "avg_fg_pct",
        "avg_ft_pct",
        "std_fp",
        "min_fp",
        "max_fp",
    ]

    # Filter to only columns that exist and have at least some non-null data
    available_columns = [
        col
        for col in display_columns
        if col in team_df.columns and (team_df[col].notna().any() if len(team_df) > 0 else False)
    ]

    # Rename columns for better display
    rename_dict = {
        "player_name": "Player",
        "health_status": "Health Status",
        "health_note": "Health Note",
        "games_played": "Games",
        "avg_fp": "Avg FP",
        "avg_points": "PTS",
        "avg_rebounds": "REB",
        "avg_assists": "AST",
        "avg_steals": "STL",
        "avg_blocks": "BLK",
        "avg_turnovers": "TOV",
        "avg_minutes": "MIN",
        "avg_fg_pct": "FG%",
        "avg_ft_pct": "FT%",
        "std_fp": "FP Std Dev",
        "min_fp": "Min FP",
        "max_fp": "Max FP",
    }

    display_df = team_df[available_columns].copy()
    display_df = display_df.rename(columns=rename_dict)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_box_plot(
    team1_players, team2_players, df_fp, date_col_trade, start_date_filter, end_date_filter
):
    """Render the fantasy points distribution box plot."""
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


def _render_trade_rating(
    team1_stats, team2_stats, team1_total_avg, team2_total_avg, team1_total_std, team2_total_std
):
    """Render the trade rating section."""
    st.markdown("---")
    st.subheader("Trade Rating")

    # Calculate rating based on multiple factors
    rating_score = 0
    rating_factors = []

    # Factor 1: Average FP difference (weighted heavily)
    if team1_total_avg > 0 and team2_total_avg > 0:
        fp_diff_pct = (
            (team1_total_avg - team2_total_avg) / max(team1_total_avg, team2_total_avg) * 100
        )
        if abs(fp_diff_pct) < 5:  # Within 5% is fair
            rating_score += 3
            rating_factors.append("✅ Average FP values are very close (fair trade)")
        elif abs(fp_diff_pct) < 10:  # Within 10% is reasonable
            rating_score += 2
            rating_factors.append("⚠️ Average FP values differ by ~10% (slight imbalance)")
        elif abs(fp_diff_pct) < 20:  # Within 20% is questionable
            rating_score += 1
            rating_factors.append("⚠️ Average FP values differ significantly (~20%)")
        else:  # More than 20% difference
            rating_score += 0
            rating_factors.append("❌ Large difference in average FP values (>20%)")

    # Factor 2: Consistency (lower std dev is better)
    if team1_total_std > 0 and team2_total_std > 0:
        std_ratio = min(team1_total_std, team2_total_std) / max(team1_total_std, team2_total_std)
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


def _render_additional_insights(team1_stats, team2_stats, team1_total_avg, team2_total_avg):
    """Render additional insights and recommendation."""
    # Additional insights
    st.markdown("---")
    st.subheader("Additional Insights")

    if team1_stats and team2_stats:
        # Best and worst players
        team1_best = max(team1_stats, key=lambda x: x["avg_fp"])
        team2_best = max(team2_stats, key=lambda x: x["avg_fp"])

        st.write("**Highest Average FP:**")
        st.write(f"- Team 1: {team1_best['player_name']} ({team1_best['avg_fp']:.2f} FP/game)")
        st.write(f"- Team 2: {team2_best['player_name']} ({team2_best['avg_fp']:.2f} FP/game)")

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

    diff = team1_total_avg - team2_total_avg
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


def _render_ai_trade_analysis(
    team1_stats,
    team2_stats,
    team1_players,
    team2_players,
    team1_total_avg,
    team2_total_avg,
    team1_total_std,
    team2_total_std,
    injury_scraper,
):
    """Render the AI Trade Analysis section."""
    st.markdown("---")
    st.subheader("🤖 AI Trade Analysis")
    st.markdown(
        "Get AI-powered insights comparing both teams, assessing strengths/weaknesses, and strategic considerations."
    )

    # Check if AI is available
    if not LANGCHAIN_AVAILABLE:
        st.warning(
            "⚠️ AI analysis requires additional packages. Install with: `pip install langchain-openai langchain`"
        )
        st.info(
            "Once installed, set your `OPENAI_API_KEY` in your `.env` file to enable AI analysis."
        )
    else:
        # Initialize chatbot in session state for trade analysis
        needs_reinit = False
        if "trade_chatbot" not in st.session_state:
            needs_reinit = True
        elif hasattr(st.session_state.trade_chatbot, "chain"):
            needs_reinit = True
        elif not hasattr(st.session_state.trade_chatbot, "llm"):
            needs_reinit = True

        if needs_reinit:
            try:
                st.session_state.trade_chatbot = PlayerAnalysisChatbot()
            except ValueError as e:
                st.error(f"❌ Configuration Error: {str(e)}")
                st.info(
                    "Please set your `OPENAI_API_KEY` in your `.env` file to enable AI analysis."
                )
            except Exception as e:
                st.error(f"❌ Error initializing AI chatbot: {str(e)}")

        # Generate AI analysis button
        if "trade_chatbot" in st.session_state:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(
                    "🤖 Generate AI Trade Analysis",
                    type="primary",
                    key="generate_trade_ai",
                    use_container_width=True,
                ):
                    with st.spinner("🤖 Analyzing trade proposal..."):
                        trade_analysis = st.session_state.trade_chatbot.generate_trade_analysis(
                            team1_stats=team1_stats,
                            team2_stats=team2_stats,
                            team1_players=team1_players,
                            team2_players=team2_players,
                            team1_total_avg=team1_total_avg,
                            team2_total_avg=team2_total_avg,
                            team1_total_std=team1_total_std,
                            team2_total_std=team2_total_std,
                            injury_scraper=injury_scraper,
                        )
                        # Store analysis in session state
                        st.session_state.trade_analysis = trade_analysis
                        st.session_state.trade_analysis_generated = True
                        st.rerun()

            with col2:
                if st.button("🔄 Clear Analysis", key="clear_trade_ai", use_container_width=True):
                    if "trade_analysis" in st.session_state:
                        del st.session_state.trade_analysis
                    if "trade_analysis_generated" in st.session_state:
                        del st.session_state.trade_analysis_generated
                    st.rerun()

            # Display AI analysis if generated
            if (
                "trade_analysis_generated" in st.session_state
                and st.session_state.trade_analysis_generated
            ):
                if "trade_analysis" in st.session_state:
                    st.markdown("---")
                    st.markdown("### 📊 AI Trade Analysis Report")
                    st.markdown(st.session_state.trade_analysis)

                    # Chat interface for follow-up questions about the trade
                    st.markdown("---")
                    st.markdown("### 💬 Ask Follow-up Questions About This Trade")

                    # Initialize trade chat messages if not exists
                    if "trade_chat_messages" not in st.session_state:
                        st.session_state.trade_chat_messages = []

                    # Display chat messages
                    for message in st.session_state.trade_chat_messages:
                        if message["role"] == "user":
                            with st.chat_message("user"):
                                st.write(message["content"])
                        else:
                            with st.chat_message("assistant"):
                                st.write(message["content"])

                    # Chat input for trade questions
                    trade_context = f"Team 1 is trading: {', '.join(team1_players) if team1_players else 'None'}. Team 2 is trading: {', '.join(team2_players) if team2_players else 'None'}."
                    if prompt := st.chat_input(
                        "Ask about this trade proposal, player comparisons, or strategic advice..."
                    ):
                        # Add user message to history
                        st.session_state.trade_chat_messages.append(
                            {"role": "user", "content": prompt}
                        )

                        # Show typing indicator
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                # Add context about the current trade
                                contextual_prompt = f"Context: {trade_context}\n\nTrade Analysis Summary:\n{st.session_state.trade_analysis[:500]}...\n\nUser Question: {prompt}"
                                bot_response = st.session_state.trade_chatbot.get_response(
                                    contextual_prompt
                                )
                                st.session_state.trade_chat_messages.append(
                                    {"role": "assistant", "content": bot_response}
                                )

                        # Rerun to display new messages
                        st.rerun()

                    # Helpful suggestions
                    st.markdown("---")
                    st.markdown("**💡 Try asking:**")
                    st.markdown("- Which side benefits more from this trade?")
                    st.markdown("- What are the biggest risks for each team?")
                    st.markdown("- Should I counter this trade? If so, what should I ask for?")
                    st.markdown("- How does this trade affect my roster depth?")
                    st.markdown("- Which players have the most upside/downside?")
