import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from injury_scraper import ESPNInjuryScraper
from style import apply_custom_css, apply_game_log_styling

# Load environment variables
load_dotenv()

# Try to import langchain and openai, but make it optional
LANGCHAIN_AVAILABLE = False
LANGCHAIN_ERROR = None

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_ERROR = str(e)
    LANGCHAIN_AVAILABLE = False

st.set_page_config(
    page_title="Fantasy Points Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for fantasy stat tracker look and feel
apply_custom_css()

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
df_fp = pd.read_csv("data/PlayerStatistics_transformed_post_21.csv")
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


# Helper function to get player personId from player name
def get_player_person_id(player_name, df_fp):
    """Get the personId (NBA player ID) for a given player name"""
    if player_name is None:
        return None
    player_data = df_fp[df_fp["player_name"] == player_name]
    if len(player_data) > 0 and "personId" in player_data.columns:
        person_id = player_data["personId"].iloc[0]
        if pd.notna(person_id):
            return int(person_id)
    return None


# Helper function to get player image URL
def get_player_image_url(person_id):
    """Get the NBA.com CDN URL for a player's headshot image"""
    if person_id is None:
        return None
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{person_id}.png"


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
        # Additional statistical averages
        "avg_points": player_data["points"].mean() if "points" in player_data.columns else None,
        "avg_rebounds": (
            player_data["reboundsTotal"].mean() if "reboundsTotal" in player_data.columns else None
        ),
        "avg_assists": player_data["assists"].mean() if "assists" in player_data.columns else None,
        "avg_steals": player_data["steals"].mean() if "steals" in player_data.columns else None,
        "avg_blocks": player_data["blocks"].mean() if "blocks" in player_data.columns else None,
        "avg_turnovers": (
            player_data["turnovers"].mean() if "turnovers" in player_data.columns else None
        ),
        "avg_minutes": (
            player_data["numMinutes"].mean() if "numMinutes" in player_data.columns else None
        ),
        "avg_fg_pct": (
            player_data["fieldGoalsPercentage"].mean()
            if "fieldGoalsPercentage" in player_data.columns
            else None
        ),
        "avg_ft_pct": (
            player_data["freeThrowsPercentage"].mean()
            if "freeThrowsPercentage" in player_data.columns
            else None
        ),
    }

    return stats


# AI Chatbot class for player performance analysis
class PlayerAnalysisChatbot:
    """AI chatbot that analyzes player performance for Yahoo Fantasy Basketball managers"""

    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain and OpenAI packages are required. Install with: pip install langchain-openai"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. Please set it in your .env file."
            )

        # Initialize OpenAI client
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key,
        )

        # System context for fantasy basketball analysis
        self.system_context = """You are an expert Yahoo Fantasy Basketball analyst helping managers make informed decisions about player value, trades, and roster management.

Your role is to:
1. Analyze player performance statistics and game logs
2. Provide actionable insights for fantasy managers
3. Evaluate player value in the context of Yahoo Fantasy Basketball scoring
4. Identify trends, consistency, and potential concerns
5. Give buy/sell/hold recommendations when appropriate

Format your analysis similar to this structure:
- Player Profile Note with key performance metrics
- Key Strengths (consistency, ceiling games, stat diversity)
- Fantasy Impact (floor, ceiling, matchup-proof status)
- Buy/Sell/Hold Recommendation
- Notes for Managers (expectations, usage, roster fit)

Be specific with numbers, cite recent performance, and provide context for fantasy managers."""

        # Initialize conversation history (manual memory management for langchain 1.x compatibility)
        self.chat_history = []

    def generate_player_summary(self, player_name, player_data, aggregated_stats, injury_info=None):
        """Generate a comprehensive player performance summary"""
        if not LANGCHAIN_AVAILABLE:
            return "AI analysis requires langchain-openai package. Install with: pip install langchain-openai"

        try:
            # Prepare player statistics summary
            recent_games = player_data.tail(22) if len(player_data) >= 22 else player_data

            # Calculate key metrics
            avg_fp = recent_games["FP"].mean()
            std_fp = recent_games["FP"].std()
            min_fp = recent_games["FP"].min()
            max_fp = recent_games["FP"].max()
            median_fp = recent_games["FP"].median()

            # Count games under certain thresholds
            games_under_50 = (recent_games["FP"] < 50).sum()
            games_over_60 = (recent_games["FP"] >= 60).sum()
            games_over_70 = (recent_games["FP"] >= 70).sum()

            # Get top performances
            top_games = recent_games.nlargest(3, "FP")[
                ["FP", "points", "reboundsTotal", "assists", "steals", "blocks"]
            ]

            # Calculate averages for key stats
            avg_points = recent_games["points"].mean()
            avg_rebounds = recent_games["reboundsTotal"].mean()
            avg_assists = recent_games["assists"].mean()
            avg_steals = recent_games["steals"].mean()
            avg_blocks = recent_games["blocks"].mean()
            avg_minutes = (
                recent_games["numMinutes"].mean() if "numMinutes" in recent_games.columns else None
            )

            # Build statistics summary text
            stats_summary = f"""
Player: {player_name}
Games Analyzed: {len(recent_games)} games

Fantasy Points Statistics:
- Average FP: {avg_fp:.1f}
- Median FP: {median_fp:.1f}
- Standard Deviation: {std_fp:.1f}
- Min FP: {min_fp:.1f}
- Max FP: {max_fp:.1f}
- Games under 50 FP: {games_under_50}
- Games 60+ FP: {games_over_60}
- Games 70+ FP: {games_over_70}

Key Statistical Averages:
- Points: {avg_points:.1f}
- Rebounds: {avg_rebounds:.1f}
- Assists: {avg_assists:.1f}
- Steals: {avg_steals:.1f}
- Blocks: {avg_blocks:.1f}
"""

            if avg_minutes:
                stats_summary += f"- Minutes: {avg_minutes:.1f}\n"

            stats_summary += "\nTop 3 Performances:\n"
            for idx, (_, game) in enumerate(top_games.iterrows(), 1):
                stats_summary += f"{idx}. {game['FP']:.1f} FP ({game['points']:.0f} pts, {game['reboundsTotal']:.0f} reb, {game['assists']:.0f} ast)\n"

            # Add injury information if available
            if injury_info:
                stats_summary += f"\nInjury Status: {injury_info.get('status', 'Unknown')}\n"
                if injury_info.get("comment"):
                    stats_summary += f"Injury Details: {injury_info.get('comment')}\n"

            # Add aggregated stats by time window if available
            if aggregated_stats is not None and len(aggregated_stats) > 0:
                stats_summary += "\nAggregated Averages by Time Window:\n"
                for _, row in aggregated_stats.iterrows():
                    stats_summary += f"- {row['Time Window']}: {row['Fantasy Points']:.1f} FP\n"

            # Create the prompt for the AI
            analysis_prompt = f"""Analyze the following player statistics and provide a comprehensive fantasy basketball analysis for Yahoo Fantasy managers.

{stats_summary}

Please provide:
1. A Player Profile Note summarizing their current performance level
2. Key Strengths (consistency, ceiling games, stat diversity, usage)
3. Fantasy Impact (floor, ceiling, matchup-proof status, roster value)
4. Buy/Sell/Hold Recommendation with reasoning
5. Notes for Managers (what to expect, usage patterns, roster fit considerations)

Format your response in a clear, actionable way that helps fantasy managers make informed decisions."""

            # Build messages with system context and analysis prompt
            messages = [
                SystemMessage(content=self.system_context),
                HumanMessage(content=analysis_prompt),
            ]

            # Get response from LLM
            response = self.llm.invoke(messages)

            # Extract content from response
            if hasattr(response, "content"):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)

            return response_text.strip()

        except Exception as e:
            return f"Error generating player summary: {str(e)}"

    def get_response(self, user_input):
        """Generate AI-powered response using LangChain and OpenAI"""
        if not LANGCHAIN_AVAILABLE:
            return "AI analysis requires langchain-openai package. Install with: pip install langchain-openai"

        try:
            # Build messages list with system context, chat history, and new user input
            messages = [SystemMessage(content=self.system_context)]

            # Add chat history
            for msg in self.chat_history:
                messages.append(msg)

            # Add current user input
            messages.append(HumanMessage(content=user_input))

            # Get response from LLM
            response = self.llm.invoke(messages)

            # Extract content from response (handles different response formats)
            if hasattr(response, "content"):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)

            # Update chat history
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=response_text))

            # Keep only last 10 exchanges to avoid token limits
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]

            return response_text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_trade_analysis(
        self,
        team1_stats,
        team2_stats,
        team1_players,
        team2_players,
        team1_total_avg,
        team2_total_avg,
        team1_total_std,
        team2_total_std,
        injury_scraper=None,
    ):
        """Generate comprehensive AI-powered trade analysis comparing both teams"""
        if not LANGCHAIN_AVAILABLE:
            return "AI analysis requires langchain-openai package. Install with: pip install langchain-openai"

        try:
            # Build detailed trade summary
            trade_summary = """
TRADE PROPOSAL ANALYSIS

TEAM 1 - Players to Trade Away:
"""
            for i, stats in enumerate(team1_stats, 1):
                # Get injury info if available
                injury_info = None
                if injury_scraper:
                    try:
                        injury_info = injury_scraper.get_player_injury(stats["player_name"])
                    except Exception:
                        pass

                trade_summary += f"""
{i}. {stats['player_name']}:
   - Average FP: {stats['avg_fp']:.2f}
   - Std Dev: {stats['std_fp']:.2f}
   - Total FP: {stats['total_fp']:.2f}
   - Games Played: {stats['games_played']}
   - Min FP: {stats['min_fp']:.2f}
   - Max FP: {stats['max_fp']:.2f}
   - Median FP: {stats['median_fp']:.2f}"""
                if injury_info:
                    trade_summary += f"\n   - Injury Status: {injury_info.get('status', 'Unknown')}"
                    if injury_info.get("comment"):
                        trade_summary += (
                            f"\n   - Injury Details: {injury_info.get('comment')[:100]}"
                        )

            trade_summary += f"""

TEAM 1 TOTALS:
- Total Average FP: {team1_total_avg:.2f}
- Combined Std Dev: {team1_total_std:.2f}
- Total Games: {sum(s['games_played'] for s in team1_stats) if team1_stats else 0}

TEAM 2 - Players to Trade For:
"""
            for i, stats in enumerate(team2_stats, 1):
                # Get injury info if available
                injury_info = None
                if injury_scraper:
                    try:
                        injury_info = injury_scraper.get_player_injury(stats["player_name"])
                    except Exception:
                        pass

                trade_summary += f"""
{i}. {stats['player_name']}:
   - Average FP: {stats['avg_fp']:.2f}
   - Std Dev: {stats['std_fp']:.2f}
   - Total FP: {stats['total_fp']:.2f}
   - Games Played: {stats['games_played']}
   - Min FP: {stats['min_fp']:.2f}
   - Max FP: {stats['max_fp']:.2f}
   - Median FP: {stats['median_fp']:.2f}"""
                if injury_info:
                    trade_summary += f"\n   - Injury Status: {injury_info.get('status', 'Unknown')}"
                    if injury_info.get("comment"):
                        trade_summary += (
                            f"\n   - Injury Details: {injury_info.get('comment')[:100]}"
                        )

            trade_summary += f"""

TEAM 2 TOTALS:
- Total Average FP: {team2_total_avg:.2f}
- Combined Std Dev: {team2_total_std:.2f}
- Total Games: {sum(s['games_played'] for s in team2_stats) if team2_stats else 0}

TRADE COMPARISON:
- FP Difference (Team 1 - Team 2): {team1_total_avg - team2_total_avg:.2f}
- FP Difference Percentage: {((team1_total_avg - team2_total_avg) / max(team1_total_avg, team2_total_avg) * 100) if max(team1_total_avg, team2_total_avg) > 0 else 0:.2f}%
"""

            # Calculate additional metrics
            if team1_stats:
                team1_avg_std = np.mean([s["std_fp"] for s in team1_stats])
                team1_best = max(team1_stats, key=lambda x: x["avg_fp"])
                team1_worst = min(team1_stats, key=lambda x: x["avg_fp"])
                trade_summary += f"""
TEAM 1 DETAILS:
- Average Consistency (Std Dev): {team1_avg_std:.2f}
- Best Player: {team1_best['player_name']} ({team1_best['avg_fp']:.2f} FP/game)
- Worst Player: {team1_worst['player_name']} ({team1_worst['avg_fp']:.2f} FP/game)
"""

            if team2_stats:
                team2_avg_std = np.mean([s["std_fp"] for s in team2_stats])
                team2_best = max(team2_stats, key=lambda x: x["avg_fp"])
                team2_worst = min(team2_stats, key=lambda x: x["avg_fp"])
                trade_summary += f"""
TEAM 2 DETAILS:
- Average Consistency (Std Dev): {team2_avg_std:.2f}
- Best Player: {team2_best['player_name']} ({team2_best['avg_fp']:.2f} FP/game)
- Worst Player: {team2_worst['player_name']} ({team2_worst['avg_fp']:.2f} FP/game)
"""

            # Create the prompt for the AI
            analysis_prompt = f"""You are an expert Yahoo Fantasy Basketball trade analyst. Analyze the following trade proposal and provide a comprehensive assessment.

{trade_summary}

Please provide a detailed trade analysis that includes:

1. **Trade Value Assessment**: Compare the overall value of both sides. Is this trade fair, or does one side have a clear advantage?

2. **Team 1 Strengths & Weaknesses**:
   - What are the strengths of the players Team 1 is giving up?
   - What are the weaknesses or concerns?
   - Who is the best/worst player in this package?

3. **Team 2 Strengths & Weaknesses**:
   - What are the strengths of the players Team 2 is giving up?
   - What are the weaknesses or concerns?
   - Who is the best/worst player in this package?

4. **Consistency Analysis**: Compare the consistency (standard deviation) of both sides. Which side has more reliable players?

5. **Risk Assessment**: Identify any risks for either side (injuries, sample size, volatility, etc.)

6. **Strategic Considerations**:
   - Which side benefits more from this trade?
   - What roster needs might this trade address?
   - Are there any positional or statistical category implications?

7. **Final Recommendation**: Should Team 1 accept, reject, or counter this trade? Provide clear reasoning.

Format your response in a clear, structured way that helps fantasy managers make informed decisions. Be specific with numbers and cite the statistics provided."""

            # Build messages with system context and analysis prompt
            trade_system_context = """You are an expert Yahoo Fantasy Basketball trade analyst with deep knowledge of player value, roster construction, and fantasy strategy.

Your role is to:
1. Objectively compare trade proposals between two teams
2. Assess the strengths and weaknesses of each side
3. Evaluate consistency, risk, and strategic value
4. Provide actionable recommendations for fantasy managers

Be analytical, specific with numbers, and help managers understand not just whether a trade is fair, but WHY and what factors to consider."""

            messages = [
                SystemMessage(content=trade_system_context),
                HumanMessage(content=analysis_prompt),
            ]

            # Get response from LLM
            response = self.llm.invoke(messages)

            # Extract content from response
            if hasattr(response, "content"):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)

            return response_text.strip()

        except Exception as e:
            return f"Error generating trade analysis: {str(e)}"

    def clear_memory(self):
        """Clear conversation memory"""
        self.chat_history = []


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

    # Display player image if available
    if player_name:
        person_id = get_player_person_id(player_name, df_fp)
        if person_id:
            image_url = get_player_image_url(person_id)
            if image_url:
                st.image(image_url, width=150, use_container_width=False)

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

            game_log, ai_analysis = st.tabs(["Game Log", "🤖 AI Analysis (beta)"])
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

                # Apply styling to game log dataframe
                styled_df = apply_game_log_styling(display_df, decimal_columns)
                st.dataframe(styled_df, width="stretch")
                # st.dataframe(player_data_ytd, use_container_width=True)
                # with stats:
                # st.subheader("Stats")
                # st.dataframe(player_data_ytd, use_container_width=True)

                # Display aggregated averages as a single table
                st.subheader("Averages")
                if len(aggregated_df) > 0:
                    st.dataframe(aggregated_df, width="stretch")

            with ai_analysis:
                st.subheader("🤖 AI Player Performance Analysis")
                st.markdown(
                    "Get AI-powered insights about this player's performance and fantasy value for Yahoo Fantasy Basketball managers."
                )

                # Check if AI is available
                if not LANGCHAIN_AVAILABLE:
                    import sys

                    python_path = sys.executable
                    venv_detected = "venv" in python_path or ".venv" in python_path

                    st.warning(
                        "⚠️ AI analysis requires additional packages. Install with: `pip install langchain-openai langchain`"
                    )

                    if LANGCHAIN_ERROR:
                        with st.expander("🔍 Debug Information"):
                            st.code(f"Import Error: {LANGCHAIN_ERROR}", language="text")
                            st.code(f"Python Path: {python_path}", language="text")
                            st.code(f"Virtual Env Detected: {venv_detected}", language="text")

                    if not venv_detected:
                        st.error(
                            "🔴 **Virtual Environment Not Active!**\n\n"
                            "It looks like you have a `venv` directory with the packages installed, but Streamlit is running "
                            "with a different Python interpreter. Please:\n\n"
                            "1. Activate your virtual environment:\n"
                            "   ```bash\n"
                            "   source venv/bin/activate  # On macOS/Linux\n"
                            "   # or\n"
                            "   venv\\Scripts\\activate  # On Windows\n"
                            "   ```\n\n"
                            "2. Then run Streamlit again:\n"
                            "   ```bash\n"
                            "   streamlit run src/FP_analyzer/app.py\n"
                            "   ```"
                        )
                    else:
                        st.info(
                            "Once installed, set your `OPENAI_API_KEY` in your `.env` file to enable AI analysis."
                        )
                else:
                    # Initialize chatbot in session state
                    # Check if chatbot exists and is valid (has new structure without 'chain')
                    needs_reinit = False
                    if "player_chatbot" not in st.session_state:
                        needs_reinit = True
                    elif hasattr(st.session_state.player_chatbot, "chain"):
                        # Old version with 'chain' attribute - needs reinitialization
                        needs_reinit = True
                        st.info("🔄 Updating AI chatbot to new version...")
                    elif not hasattr(st.session_state.player_chatbot, "llm"):
                        # Missing required attributes - needs reinitialization
                        needs_reinit = True

                    if needs_reinit:
                        try:
                            st.session_state.player_chatbot = PlayerAnalysisChatbot()
                            if "ai_chat_messages" not in st.session_state:
                                st.session_state.ai_chat_messages = []
                        except ValueError as e:
                            st.error(f"❌ Configuration Error: {str(e)}")
                            st.info(
                                "Please set your `OPENAI_API_KEY` in your `.env` file to enable AI analysis."
                            )
                            st.stop()
                        except Exception as e:
                            st.error(f"❌ Error initializing AI chatbot: {str(e)}")
                            st.stop()

                    # Get injury info for the summary
                    try:
                        injury_info = injury_scraper.get_player_injury(player_name)
                    except Exception:
                        injury_info = None

                    # Generate summary button
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.button(
                            "📊 Generate Player Performance Summary",
                            type="primary",
                            use_container_width=True,
                        ):
                            with st.spinner("🤖 Analyzing player performance..."):
                                summary = st.session_state.player_chatbot.generate_player_summary(
                                    player_name=player_name,
                                    player_data=player_data_ytd,
                                    aggregated_stats=aggregated_df,
                                    injury_info=injury_info,
                                )
                                # Store summary in session state
                                st.session_state.player_summary = summary
                                st.session_state.summary_generated = True

                    with col2:
                        if st.button("🔄 Clear Chat", use_container_width=True):
                            st.session_state.ai_chat_messages = []
                            st.session_state.player_chatbot.clear_memory()
                            if "player_summary" in st.session_state:
                                del st.session_state.player_summary
                            if "summary_generated" in st.session_state:
                                del st.session_state.summary_generated
                            st.rerun()

                    # Display summary if generated
                    if (
                        "summary_generated" in st.session_state
                        and st.session_state.summary_generated
                    ):
                        if "player_summary" in st.session_state:
                            st.markdown("---")
                            st.markdown("### 📝 Player Performance Summary")
                            st.markdown(st.session_state.player_summary)

                    # Chat interface for follow-up questions
                    st.markdown("---")
                    st.markdown("### 💬 Ask Follow-up Questions")

                    # Display chat messages
                    for message in st.session_state.ai_chat_messages:
                        if message["role"] == "user":
                            with st.chat_message("user"):
                                st.write(message["content"])
                        else:
                            with st.chat_message("assistant"):
                                st.write(message["content"])

                    # Chat input
                    if prompt := st.chat_input(
                        f"Ask about {player_name}'s fantasy value, trends, or trade advice..."
                    ):
                        # Add user message to history
                        st.session_state.ai_chat_messages.append(
                            {"role": "user", "content": prompt}
                        )

                        # Show typing indicator
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                # Add context about the current player
                                contextual_prompt = (
                                    f"Context: We're analyzing {player_name}. {prompt}"
                                )
                                bot_response = st.session_state.player_chatbot.get_response(
                                    contextual_prompt
                                )
                                st.session_state.ai_chat_messages.append(
                                    {"role": "assistant", "content": bot_response}
                                )

                        # Rerun to display new messages
                        st.rerun()

                    # Helpful suggestions
                    st.markdown("---")
                    st.markdown("**💡 Try asking:**")
                    st.markdown(f"- What's {player_name}'s consistency like?")
                    st.markdown(f"- Should I buy, sell, or hold {player_name}?")
                    st.markdown(f"- What's {player_name}'s fantasy ceiling and floor?")
                    st.markdown(
                        f"- How does {player_name} compare to other players at their position?"
                    )
                    st.markdown(f"- What are the concerns about {player_name}'s performance?")

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
            st.markdown("---")
            st.subheader("Player Breakdown")

            breakdown_col1, breakdown_col2 = st.columns(2)

            with breakdown_col1:
                st.write("**Team 1 Players:**")
                if team1_stats:
                    # Fetch injury info for each player
                    for stats in team1_stats:
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

                    team1_df = pd.DataFrame(team1_stats)
                    team1_df = team1_df.round(2)

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
                        if col in team1_df.columns
                        and (team1_df[col].notna().any() if len(team1_df) > 0 else False)
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

                    display_df = team1_df[available_columns].copy()
                    display_df = display_df.rename(columns=rename_dict)

                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.write("No players selected")

            with breakdown_col2:
                st.write("**Team 2 Players:**")
                if team2_stats:
                    # Fetch injury info for each player
                    for stats in team2_stats:
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

                    team2_df = pd.DataFrame(team2_stats)
                    team2_df = team2_df.round(2)

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
                        if col in team2_df.columns
                        and (team2_df[col].notna().any() if len(team2_df) > 0 else False)
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

                    display_df = team2_df[available_columns].copy()
                    display_df = display_df.rename(columns=rename_dict)

                    st.dataframe(display_df, use_container_width=True, hide_index=True)
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
                    rating_factors.append("⚠️ Average FP values differ by ~10% (slight imbalance)")
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

            # AI Trade Analysis
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
                                trade_analysis = (
                                    st.session_state.trade_chatbot.generate_trade_analysis(
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
                                )
                                # Store analysis in session state
                                st.session_state.trade_analysis = trade_analysis
                                st.session_state.trade_analysis_generated = True
                                st.rerun()

                    with col2:
                        if st.button(
                            "🔄 Clear Analysis", key="clear_trade_ai", use_container_width=True
                        ):
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
                            st.markdown(
                                "- Should I counter this trade? If so, what should I ask for?"
                            )
                            st.markdown("- How does this trade affect my roster depth?")
                            st.markdown("- Which players have the most upside/downside?")
