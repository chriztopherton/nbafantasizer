"""NBA Stats fetching and display utilities using nba_api."""

import pandas as pd
import streamlit as st
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import playercareerstats


@st.cache_data(ttl=60)  # Cache for 1 minute (live data changes frequently)
def get_live_scoreboard():
    """
    Get today's live scoreboard from NBA API.

    Returns:
        dict or None: Scoreboard data as dictionary if successful, None otherwise.
    """
    try:
        games = scoreboard.ScoreBoard()
        return games.get_dict()
    except Exception:
        # Silently fail for live data - it's okay if it's not available
        return None


def get_player_live_stats(player_id, scoreboard_data=None):
    """
    Get live stats for a player from today's games.

    Args:
        player_id (int or str): NBA player ID.
        scoreboard_data (dict, optional): Pre-fetched scoreboard data.

    Returns:
        dict or None: Live stats for the player if playing, None otherwise.
    """
    if player_id is None:
        return None

    # Fetch scoreboard if not provided
    if scoreboard_data is None:
        scoreboard_data = get_live_scoreboard()

    if scoreboard_data is None:
        return None

    try:
        # Navigate through the scoreboard structure to find player stats
        # The structure can vary, so we'll try multiple paths
        games = None

        # Try different possible structures
        if "scoreboard" in scoreboard_data:
            if isinstance(scoreboard_data["scoreboard"], dict):
                games = scoreboard_data["scoreboard"].get("games", [])
            elif isinstance(scoreboard_data["scoreboard"], list):
                games = scoreboard_data["scoreboard"]
        elif "games" in scoreboard_data:
            games = scoreboard_data["games"]

        if not games:
            return None

        for game in games:
            # Check home team players
            home_team = game.get("homeTeam", {})
            if not home_team:
                continue

            home_players = home_team.get("players", [])
            if not home_players:
                # Try alternative structure
                home_players = home_team.get("playerStats", [])

            for player in home_players:
                player_id_val = player.get("personId") or player.get("playerId") or player.get("id")
                if str(player_id_val) == str(player_id):
                    return {
                        "player": player,
                        "team": home_team.get("teamName")
                        or home_team.get("teamTricode", "Unknown"),
                        "opponent": game.get("awayTeam", {}).get("teamName")
                        or game.get("awayTeam", {}).get("teamTricode", "Unknown"),
                        "gameStatus": game.get("gameStatusText")
                        or game.get("gameStatus", "Unknown"),
                        "gameId": game.get("gameId", ""),
                    }

            # Check away team players
            away_team = game.get("awayTeam", {})
            if not away_team:
                continue

            away_players = away_team.get("players", [])
            if not away_players:
                # Try alternative structure
                away_players = away_team.get("playerStats", [])

            for player in away_players:
                player_id_val = player.get("personId") or player.get("playerId") or player.get("id")
                if str(player_id_val) == str(player_id):
                    return {
                        "player": player,
                        "team": away_team.get("teamName")
                        or away_team.get("teamTricode", "Unknown"),
                        "opponent": game.get("homeTeam", {}).get("teamName")
                        or game.get("homeTeam", {}).get("teamTricode", "Unknown"),
                        "gameStatus": game.get("gameStatusText")
                        or game.get("gameStatus", "Unknown"),
                        "gameId": game.get("gameId", ""),
                    }
    except Exception:
        # Silently fail - live data might not be available or structure might be different
        return None

    return None


def display_live_stats(live_stats):
    """
    Display live game stats for a player.

    Args:
        live_stats (dict): Live stats dictionary from get_player_live_stats.
    """
    if live_stats is None:
        return

    player_data = live_stats.get("player", {})
    if not player_data:
        return

    # Extract stats from player data - stats might be directly in player_data or in a statistics field
    stats = player_data.get("statistics", {})
    if not stats:
        # Try using player_data directly as stats
        stats = player_data

    st.markdown("### 🔴 Live Game Stats")

    # Game info
    team = live_stats.get("team", "Unknown")
    opponent = live_stats.get("opponent", "Unknown")
    game_status = live_stats.get("gameStatus", "Unknown")

    st.markdown(f"**{team}** vs **{opponent}** - {game_status}")

    # Extract key stats (field names may vary, so we'll try common ones)
    # Try multiple possible field name variations
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
    fgm = stats.get("fieldGoalsMade") or stats.get("fgm") or stats.get("FGM") or 0
    fga = stats.get("fieldGoalsAttempted") or stats.get("fga") or stats.get("FGA") or 0
    ftm = stats.get("freeThrowsMade") or stats.get("ftm") or stats.get("FTM") or 0
    fta = stats.get("freeThrowsAttempted") or stats.get("fta") or stats.get("FTA") or 0
    fg3m = (
        stats.get("threePointersMade")
        or stats.get("fg3m")
        or stats.get("FG3M")
        or stats.get("threePointFieldGoalsMade")
        or 0
    )

    # Calculate percentages
    fg_pct = (fgm / fga * 100) if fga and fga > 0 else 0
    ft_pct = (ftm / fta * 100) if fta and fta > 0 else 0

    # Display stats in columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("PTS", format_stat_value(pts))
    with col2:
        st.metric("REB", format_stat_value(reb))
    with col3:
        st.metric("AST", format_stat_value(ast))
    with col4:
        st.metric("STL", format_stat_value(stl))
    with col5:
        st.metric("BLK", format_stat_value(blk))
    with col6:
        st.metric("TOV", format_stat_value(tov))

    # Second row
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "MIN", min_played if isinstance(min_played, str) else format_stat_value(min_played)
        )
    with col2:
        st.metric("FG%", f"{fg_pct:.1f}%" if fg_pct > 0 else "N/A")
    with col3:
        st.metric("FT%", f"{ft_pct:.1f}%" if ft_pct > 0 else "N/A")
    with col4:
        st.metric("3PTM", format_stat_value(fg3m))
    with col5:
        st.metric("FGM/A", f"{fgm}/{fga}" if fga and fga > 0 else "N/A")
    with col6:
        st.metric("FTM/A", f"{ftm}/{fta}" if fta and fta > 0 else "N/A")


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_player_career_stats(player_id):
    """
    Get player career statistics from NBA API.

    Args:
        player_id (int or str): NBA player ID.

    Returns:
        pd.DataFrame or None: Career stats DataFrame if successful, None otherwise.
    """
    if player_id is None:
        return None

    try:
        career = playercareerstats.PlayerCareerStats(player_id=str(player_id))
        # Get regular season totals
        career_df = career.season_totals_regular_season.get_data_frame()
        return career_df
    except Exception as e:
        st.warning(f"Could not fetch career stats: {e}")
        return None


def get_current_season_stats(career_df):
    """
    Extract current season stats from career DataFrame and calculate per-game averages.

    Args:
        career_df (pd.DataFrame): Career stats DataFrame.

    Returns:
        pd.Series or None: Current season stats with per-game averages if available.
    """
    if career_df is None or len(career_df) == 0:
        return None

    # Get the most recent season (assuming data is sorted by season)
    # The current season would be the last row
    current_season = career_df.iloc[-1].copy()

    # Calculate per-game averages for key stats
    gp = current_season.get("GP", 0)
    if gp and pd.notna(gp) and gp > 0:
        # Calculate per-game averages
        stat_columns = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN"]
        for col in stat_columns:
            if col in current_season and pd.notna(current_season[col]):
                # Store per-game average
                current_season[f"{col}_PG"] = current_season[col] / gp

    return current_season


def format_stat_value(value, stat_type="default"):
    """
    Format a stat value for display.

    Args:
        value: The stat value to format.
        stat_type: Type of stat ('percentage', 'minutes', 'default').

    Returns:
        str: Formatted stat value.
    """
    if pd.isna(value) or value is None:
        return "N/A"

    if stat_type == "percentage":
        return f"{value:.1%}"
    elif stat_type == "minutes":
        return f"{value:.1f}"
    elif stat_type == "decimal":
        return f"{value:.2f}"
    else:
        # For integers, round and format
        if isinstance(value, int | float):
            return f"{int(round(value))}"
        return str(value)


def display_player_stats_header(career_df, current_season_stats, player_name):
    """
    Display player stats header similar to Yahoo Fantasy UI.

    Args:
        career_df (pd.DataFrame): Career stats DataFrame.
        current_season_stats (pd.Series): Current season stats.
        player_name (str): Player name.
    """
    if current_season_stats is None:
        st.warning("Current season stats not available.")
        return

    # Get season ID for display
    season_id = current_season_stats.get("SEASON_ID", "N/A")
    gp = current_season_stats.get("GP", 0)

    # Display season info
    st.markdown(f"### 📊 {season_id} Season Stats ({gp} GP)")

    # Calculate per-game averages
    if gp and gp > 0:
        pts_pg = current_season_stats.get("PTS_PG") or (current_season_stats.get("PTS", 0) / gp)
        reb_pg = current_season_stats.get("REB_PG") or (current_season_stats.get("REB", 0) / gp)
        ast_pg = current_season_stats.get("AST_PG") or (current_season_stats.get("AST", 0) / gp)
        stl_pg = current_season_stats.get("STL_PG") or (current_season_stats.get("STL", 0) / gp)
        blk_pg = current_season_stats.get("BLK_PG") or (current_season_stats.get("BLK", 0) / gp)
        tov_pg = current_season_stats.get("TOV_PG") or (current_season_stats.get("TOV", 0) / gp)
        fg3m_pg = current_season_stats.get("FG3M_PG") or (current_season_stats.get("FG3M", 0) / gp)
        min_pg = (
            current_season_stats.get("MIN", 0) / gp if current_season_stats.get("MIN", 0) else 0
        )
    else:
        pts_pg = reb_pg = ast_pg = stl_pg = blk_pg = tov_pg = fg3m_pg = min_pg = 0

    # Create columns for key metrics (per-game averages)
    st.markdown("**Per Game Averages:**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("PTS", format_stat_value(pts_pg, "decimal"))
    with col2:
        st.metric("REB", format_stat_value(reb_pg, "decimal"))
    with col3:
        st.metric("AST", format_stat_value(ast_pg, "decimal"))
    with col4:
        st.metric("STL", format_stat_value(stl_pg, "decimal"))
    with col5:
        st.metric("BLK", format_stat_value(blk_pg, "decimal"))
    with col6:
        st.metric("GP", format_stat_value(gp))

    # Second row of metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("MIN", format_stat_value(min_pg, "decimal"))
    with col2:
        fg_pct = current_season_stats.get("FG_PCT", 0)
        st.metric("FG%", format_stat_value(fg_pct, "percentage"))
    with col3:
        ft_pct = current_season_stats.get("FT_PCT", 0)
        st.metric("FT%", format_stat_value(ft_pct, "percentage"))
    with col4:
        st.metric("3PTM", format_stat_value(fg3m_pg, "decimal"))
    with col5:
        st.metric("TOV", format_stat_value(tov_pg, "decimal"))
    with col6:
        plus_minus = current_season_stats.get("PLUS_MINUS", 0)
        if gp and gp > 0:
            plus_minus_pg = plus_minus / gp
        else:
            plus_minus_pg = 0
        st.metric("+/-", format_stat_value(plus_minus_pg, "decimal"))


def display_season_history(career_df):
    """
    Display historical season-by-season stats in a table format.

    Args:
        career_df (pd.DataFrame): Career stats DataFrame.
    """
    if career_df is None or len(career_df) == 0:
        st.warning("Historical stats not available.")
        return

    st.markdown("### 📈 Season History")

    # Select relevant columns for display
    display_columns = [
        "SEASON_ID",
        "GP",
        "MIN",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FG_PCT",
        "FT_PCT",
        "FG3M",
    ]

    # Filter to only include columns that exist
    available_columns = [col for col in display_columns if col in career_df.columns]
    history_df = career_df[available_columns].copy()

    # Sort by season (most recent first)
    history_df = history_df.sort_values("SEASON_ID", ascending=False)

    # Calculate per-game averages for all seasons
    stat_columns = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]
    for col in stat_columns:
        if col in history_df.columns:
            # Calculate per-game average
            history_df[f"{col}_PG"] = history_df.apply(
                lambda row, col=col: (
                    row[col] / row["GP"] if pd.notna(row["GP"]) and row["GP"] > 0 else 0
                ),
                axis=1,
            )

    # Calculate MIN per game
    if "MIN" in history_df.columns:
        history_df["MIN_PG"] = history_df.apply(
            lambda row: row["MIN"] / row["GP"] if pd.notna(row["GP"]) and row["GP"] > 0 else 0,
            axis=1,
        )

    # Format percentage columns
    if "FG_PCT" in history_df.columns:
        history_df["FG_PCT"] = history_df["FG_PCT"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
    if "FT_PCT" in history_df.columns:
        history_df["FT_PCT"] = history_df["FT_PCT"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )

    # Format per-game averages to 2 decimal places
    pg_columns = ["PTS_PG", "REB_PG", "AST_PG", "STL_PG", "BLK_PG", "TOV_PG", "FG3M_PG", "MIN_PG"]
    for col in pg_columns:
        if col in history_df.columns:
            history_df[col] = history_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    # Keep GP as integer
    if "GP" in history_df.columns:
        history_df["GP"] = history_df["GP"].apply(
            lambda x: f"{int(round(x))}" if pd.notna(x) else "N/A"
        )

    # Select columns for final display (per-game averages)
    final_columns = [
        "SEASON_ID",
        "GP",
        "MIN_PG",
        "PTS_PG",
        "REB_PG",
        "AST_PG",
        "STL_PG",
        "BLK_PG",
        "TOV_PG",
        "FG_PCT",
        "FT_PCT",
        "FG3M_PG",
    ]

    # Filter to only include columns that exist
    final_available = [col for col in final_columns if col in history_df.columns]
    history_df = history_df[final_available].copy()

    # Rename columns for better display (only rename columns that exist)
    column_renames = {
        "SEASON_ID": "Season",
        "MIN_PG": "MIN",
        "PTS_PG": "PTS",
        "REB_PG": "REB",
        "AST_PG": "AST",
        "STL_PG": "STL",
        "BLK_PG": "BLK",
        "TOV_PG": "TOV",
        "FG_PCT": "FG%",
        "FT_PCT": "FT%",
        "FG3M_PG": "3PTM",
    }

    # Only rename columns that exist in the dataframe
    existing_renames = {k: v for k, v in column_renames.items() if k in history_df.columns}
    history_df = history_df.rename(columns=existing_renames)

    # Display the table
    st.dataframe(history_df, use_container_width=True, hide_index=True)


def render_nba_stats_section(player_id, player_name):
    """
    Render the complete NBA stats section for a player.

    Args:
        player_id (int or str): NBA player ID.
        player_name (str): Player name.
    """
    if player_id is None:
        st.warning("Player ID not available. Cannot fetch NBA stats.")
        return

    # Check for live stats first
    live_stats = get_player_live_stats(player_id)
    if live_stats:
        display_live_stats(live_stats)
        st.markdown("---")

    # Fetch career stats
    career_df = get_player_career_stats(player_id)

    if career_df is None or len(career_df) == 0:
        st.warning("Could not fetch NBA stats for this player.")
        return

    # Get current season stats
    current_season_stats = get_current_season_stats(career_df)

    # Display stats
    display_player_stats_header(career_df, current_season_stats, player_name)
    st.markdown("---")
    display_season_history(career_df)
