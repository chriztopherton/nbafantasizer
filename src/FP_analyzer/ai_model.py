"""AI model functionality for Fantasy Points Analyzer application."""

import os

import numpy as np

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


class PlayerAnalysisChatbot:
    """
    AI chatbot that analyzes player performance for Yahoo Fantasy Basketball managers.

    This class provides methods for generating AI-powered player performance summaries,
    trade analysis, and interactive chat functionality using OpenAI's language models.
    """

    def __init__(self):
        """
        Initialize the AI chatbot with OpenAI client and system context.

        Raises:
            ImportError: If LangChain packages are not available.
            ValueError: If OPENAI_API_KEY is not found in environment variables.
        """
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
        """
        Generate a comprehensive player performance summary using AI.

        Args:
            player_name (str): Name of the player to analyze.
            player_data (pd.DataFrame): DataFrame containing player game data.
            aggregated_stats (pd.DataFrame): DataFrame with aggregated statistics by time window.
            injury_info (dict, optional): Dictionary containing injury information for the player.

        Returns:
            str: AI-generated player performance summary, or error message if generation fails.
        """
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
                team = injury_info.get("team", "")
                if team:
                    stats_summary += f"Team: {team}\n"
                updated = injury_info.get("updated", "")
                if updated:
                    stats_summary += f"Last Updated: {updated}\n"
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
        """
        Generate AI-powered response using LangChain and OpenAI.

        Maintains conversation history and generates contextual responses based on
        the system context and previous messages.

        Args:
            user_input (str): User's input message/question.

        Returns:
            str: AI-generated response, or error message if generation fails.
        """
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
        """
        Generate comprehensive AI-powered trade analysis comparing both teams.

        Args:
            team1_stats (list): List of dictionaries containing stats for Team 1 players.
            team2_stats (list): List of dictionaries containing stats for Team 2 players.
            team1_players (list): List of player names for Team 1.
            team2_players (list): List of player names for Team 2.
            team1_total_avg (float): Total average FP for Team 1.
            team2_total_avg (float): Total average FP for Team 2.
            team1_total_std (float): Combined standard deviation for Team 1.
            team2_total_std (float): Combined standard deviation for Team 2.
            injury_scraper (optional): Injury scraper object for fetching player injury info.

        Returns:
            str: AI-generated trade analysis, or error message if generation fails.
        """
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
                    team = injury_info.get("team", "")
                    if team:
                        trade_summary += f"\n   - Team: {team}"
                    updated = injury_info.get("updated", "")
                    if updated:
                        trade_summary += f"\n   - Last Updated: {updated}"
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
                    team = injury_info.get("team", "")
                    if team:
                        trade_summary += f"\n   - Team: {team}"
                    updated = injury_info.get("updated", "")
                    if updated:
                        trade_summary += f"\n   - Last Updated: {updated}"
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
        """
        Clear conversation memory.

        Resets the chat history, useful when starting a new conversation or
        changing context (e.g., switching to a different player).
        """
        self.chat_history = []
