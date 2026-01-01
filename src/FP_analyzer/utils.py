"""Utility functions for Fantasy Points Analyzer application."""

import os
import zipfile
from io import BytesIO

import pandas as pd
import requests


def find_and_process_date_column(data):
    """
    Find and process the date column in a dataframe.

    Attempts to locate a date column by checking common column names, then
    converts it to datetime format, handling timezone issues.

    Args:
        data (pd.DataFrame): The dataframe to search for date columns.

    Returns:
        str or None: The name of the found date column, or None if not found.
    """
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
        except Exception:
            pass

    return date_col


def filter_data_by_date(data, date_col, start_date, end_date):
    """
    Filter dataframe by date range.

    Removes rows with missing dates or FP values, then filters to the
    specified date range and sorts by date.

    Args:
        data (pd.DataFrame): The dataframe to filter.
        date_col (str): Name of the date column.
        start_date: Start date for filtering (can be datetime, date, or string).
        end_date: End date for filtering (can be datetime, date, or string).

    Returns:
        pd.DataFrame: Filtered dataframe sorted by date.
    """
    if date_col is None or len(data) == 0:
        return data

    try:
        data = data.dropna(subset=[date_col, "FP"])
        start_date_ts = pd.Timestamp(start_date).normalize()
        end_date_ts = pd.Timestamp(end_date).normalize()
        data["date_only"] = pd.to_datetime(data[date_col]).dt.normalize()
        data = data[(data["date_only"] >= start_date_ts) & (data["date_only"] <= end_date_ts)]
        data = data.sort_values(date_col)
    except Exception:
        pass

    return data


def get_player_person_id(player_name, df_fp):
    """
    Get the personId (NBA player ID) for a given player name.

    Args:
        player_name (str): Name of the player to look up.
        df_fp (pd.DataFrame): DataFrame containing player data with 'player_name' and 'personId' columns.

    Returns:
        int or None: The personId if found, None otherwise.
    """
    if player_name is None:
        return None
    player_data = df_fp[df_fp["player_name"] == player_name]
    if len(player_data) > 0 and "personId" in player_data.columns:
        person_id = player_data["personId"].iloc[0]
        if pd.notna(person_id):
            return int(person_id)
    return None


def get_player_image_url(person_id):
    """
    Get the NBA.com CDN URL for a player's headshot image.

    Args:
        person_id (int): The NBA personId for the player.

    Returns:
        str or None: The image URL if person_id is valid, None otherwise.
    """
    if person_id is None:
        return None
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{person_id}.png"


def get_player_attributes(player_name, df_fp, injury_scraper=None):
    """
    Get player attributes: team, height, and position.

    Retrieves the most recent team name, height (converted to feet/inches if needed),
    and position from the dataframe. Optionally supplements position from injury scraper.

    Args:
        player_name (str): Name of the player to look up.
        df_fp (pd.DataFrame): DataFrame containing player data.
        injury_scraper (optional): Injury scraper object with get_player_injury method.

    Returns:
        dict or None: Dictionary with 'team', 'height', and 'position' keys, or None if player not found.
    """
    if player_name is None:
        return None

    attributes = {
        "team": None,
        "height": None,
        "position": None,
    }

    # Get team from dataframe (use most recent team)
    player_data = df_fp[df_fp["player_name"] == player_name]
    if len(player_data) > 0:
        # Get the most recent team name
        if "playerteamName" in player_data.columns:
            # Get the most recent non-null team name
            team_names = player_data["playerteamName"].dropna()
            if len(team_names) > 0:
                # Get the most recent team (assuming data is sorted by date)
                attributes["team"] = team_names.iloc[-1] if len(team_names) > 0 else None

        # Check for height column (might not exist)
        if "height" in player_data.columns:
            height_values = player_data["height"].dropna()
            if len(height_values) > 0:
                attributes["height"] = height_values.iloc[0]
        elif "heightInches" in player_data.columns:
            height_values = player_data["heightInches"].dropna()
            if len(height_values) > 0:
                height_inches = height_values.iloc[0]
                # Convert to feet and inches format
                if pd.notna(height_inches):
                    feet = int(height_inches // 12)
                    inches = int(height_inches % 12)
                    attributes["height"] = f"{feet}'{inches}\""

        # Check for position column
        if "position" in player_data.columns:
            position_values = player_data["position"].dropna()
            if len(position_values) > 0:
                attributes["position"] = position_values.iloc[0]

    # Try to get position from injury scraper if not found in dataframe
    if attributes["position"] is None and injury_scraper:
        try:
            injury_info = injury_scraper.get_player_injury(player_name)
            if injury_info and injury_info.get("position"):
                attributes["position"] = injury_info.get("position")
        except Exception:
            pass

    return attributes


def get_player_fp_data(player_name, df_fp, date_col, start_date, end_date):
    """
    Get all FP (Fantasy Points) values for a player within a date range.

    Filters player data by name and date range, then returns the FP values.

    Args:
        player_name (str): Name of the player to look up.
        df_fp (pd.DataFrame): DataFrame containing player data.
        date_col (str): Name of the date column in the dataframe.
        start_date: Start date for filtering.
        end_date: End date for filtering.

    Returns:
        np.ndarray or None: Array of FP values if data found, None otherwise.
    """
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


def calculate_player_trade_stats(player_name, df_fp, date_col, start_date, end_date):
    """
    Calculate statistics for a player for trade analysis.

    Computes fantasy points statistics (mean, std, min, max, median) and
    averages for various stat categories (points, rebounds, assists, etc.)
    within the specified date range.

    Args:
        player_name (str): Name of the player to analyze.
        df_fp (pd.DataFrame): DataFrame containing player data.
        date_col (str): Name of the date column in the dataframe.
        start_date: Start date for the analysis period.
        end_date: End date for the analysis period.

    Returns:
        dict or None: Dictionary containing player statistics including avg_fp, std_fp,
                     total_fp, games_played, min_fp, max_fp, median_fp, and various
                     stat averages. Returns None if no data found.
    """
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


def analyze_recent_games(player_data, date_col, num_games=7):
    """
    Analyze player's performance in the last N games.

    Analyzes recent game performance, calculates averages, identifies standout
    performances, detects trends, and generates a formatted summary string.

    Args:
        player_data (pd.DataFrame): DataFrame containing player game data.
        date_col (str): Name of the date column to sort by.
        num_games (int): Number of recent games to analyze (default: 7).

    Returns:
        str or None: Formatted analysis string with performance summary, or None if no data.
    """
    if len(player_data) == 0:
        return None

    # Sort by date descending to get most recent games
    sorted_data = player_data.sort_values(by=date_col, ascending=False)
    recent_data = sorted_data.head(num_games)

    if len(recent_data) == 0:
        return None

    # Calculate averages for recent games
    avg_fp = recent_data["FP"].mean()
    avg_points = recent_data["points"].mean()
    avg_rebounds = recent_data["reboundsTotal"].mean()
    avg_assists = recent_data["assists"].mean()
    avg_steals = recent_data["steals"].mean()
    avg_blocks = recent_data["blocks"].mean()
    avg_turnovers = recent_data["turnovers"].mean()
    avg_minutes = recent_data["numMinutes"].mean()
    avg_fg_pct = recent_data["fieldGoalsPercentage"].mean()

    # Compare to season averages (using all available data)
    season_avg_fp = player_data["FP"].mean()
    season_avg_fg_pct = player_data["fieldGoalsPercentage"].mean()

    # Get opponent information for recent games
    try:
        if "opponent_with_at" in recent_data.columns:
            recent_opponents = recent_data["opponent_with_at"].tolist()
        else:
            if "home" in recent_data.columns and "opponentteamName" in recent_data.columns:
                recent_opponents = [
                    f"@{opp}" if home == 0 else str(opp)
                    for opp, home in zip(
                        recent_data["opponentteamName"], recent_data["home"], strict=False
                    )
                ]
            elif "opponentteamName" in recent_data.columns:
                recent_opponents = recent_data["opponentteamName"].tolist()
            else:
                recent_opponents = None
    except Exception:
        recent_opponents = None

    # Find standout performances (use iloc to get position in sorted dataframe)
    max_fp_pos = recent_data["FP"].values.argmax()
    min_fp_pos = recent_data["FP"].values.argmin()
    max_fp_game = recent_data.iloc[max_fp_pos]
    min_fp_game = recent_data.iloc[min_fp_pos]

    # Get opponent for standout games
    max_opponent = (
        recent_opponents[max_fp_pos]
        if recent_opponents and max_fp_pos < len(recent_opponents)
        else None
    )
    min_opponent = (
        recent_opponents[min_fp_pos]
        if recent_opponents and min_fp_pos < len(recent_opponents)
        else None
    )

    # Calculate trends (compare most recent half to older half)
    if len(recent_data) >= 4:
        mid_point = len(recent_data) // 2
        most_recent_half = recent_data.iloc[:mid_point][
            "FP"
        ].mean()  # Most recent games (top of sorted list)
        older_half = recent_data.iloc[mid_point:][
            "FP"
        ].mean()  # Older games (bottom of sorted list)
        trend_diff = most_recent_half - older_half
        trend_pct = (trend_diff / older_half * 100) if older_half > 0 else 0

        if trend_pct > 5:
            fp_trend = f"improving (+{trend_pct:.1f}%)"
        elif trend_pct < -5:
            fp_trend = f"declining ({trend_pct:.1f}%)"
        else:
            fp_trend = "stable"
    else:
        fp_trend = None

    # Build analysis text
    analysis_parts = []
    analysis_parts.append(f"**Recent {len(recent_data)} Games Performance:**\n")

    # Overall summary
    fp_vs_season = ((avg_fp - season_avg_fp) / season_avg_fp * 100) if season_avg_fp > 0 else 0
    if fp_vs_season > 5:
        performance_summary = f"Strong performance ({fp_vs_season:.1f}% above season average)"
    elif fp_vs_season < -5:
        performance_summary = (
            f"Below average performance ({abs(fp_vs_season):.1f}% below season average)"
        )
    else:
        performance_summary = f"Consistent with season average ({fp_vs_season:+.1f}%)"

    analysis_parts.append(f"- **Overall:** {performance_summary}\n")

    # List opponents faced
    if recent_opponents is not None and len(recent_opponents) > 0:
        unique_opponents = list(
            dict.fromkeys(recent_opponents)
        )  # Preserve order while removing duplicates
        opponents_text = ", ".join(
            str(opp) for opp in unique_opponents[:5]
        )  # Show up to 5 unique opponents
        if len(unique_opponents) > 5:
            opponents_text += f" (+{len(unique_opponents) - 5} more)"
        analysis_parts.append(f"- **Opponents:** {opponents_text}\n")

    # Key stats
    analysis_parts.append(
        f"- **Averages:** {avg_fp:.1f} FP, {avg_points:.1f} PTS, {avg_rebounds:.1f} REB, {avg_assists:.1f} AST, {avg_steals:.1f} STL, {avg_blocks:.1f} BLK\n"
    )

    # Standout performances
    if max_fp_game["FP"] > season_avg_fp * 1.2:
        opponent_text = f" vs {max_opponent}" if max_opponent else ""
        analysis_parts.append(
            f"- **Standout Game:** {max_fp_game['FP']:.1f} FP ({max_fp_game['points']:.0f} PTS, {max_fp_game['reboundsTotal']:.0f} REB, {max_fp_game['assists']:.0f} AST){opponent_text}\n"
        )

    # What's worth noting
    notable_items = []
    if max_fp_game["FP"] > avg_fp * 1.5:
        notable_items.append(f"exceptional {max_fp_game['FP']:.1f} FP game")
    if avg_steals > 2.0:
        notable_items.append("strong steals production")
    if avg_blocks > 2.0:
        notable_items.append("strong shot-blocking")
    if avg_assists > 7.0:
        notable_items.append("high assist numbers")
    if avg_rebounds > 10.0:
        notable_items.append("strong rebounding")

    if notable_items:
        analysis_parts.append(f"- **Notable:** {', '.join(notable_items)}\n")

    # Struggles/Concerns
    struggles = []
    if avg_fp < season_avg_fp * 0.85:
        struggles.append("lower fantasy production")
    if avg_turnovers > 3.5:
        struggles.append(f"high turnovers ({avg_turnovers:.1f} per game)")
    if avg_fg_pct < 0.40 and season_avg_fg_pct > 0.45:
        struggles.append(f"poor shooting efficiency ({avg_fg_pct:.1%} FG%)")
    if avg_minutes < 25 and len(player_data) > 5:
        avg_season_minutes = player_data["numMinutes"].mean()
        if avg_minutes < avg_season_minutes * 0.9:
            struggles.append(f"reduced minutes ({avg_minutes:.1f} MPG)")
    if min_fp_game["FP"] < season_avg_fp * 0.6 and len(recent_data) >= 3:
        opponent_text = f" vs {min_opponent}" if min_opponent else ""
        struggles.append(f"one concerning {min_fp_game['FP']:.1f} FP game{opponent_text}")

    if struggles:
        analysis_parts.append(f"- **Concerns:** {', '.join(struggles)}\n")

    # Trends
    if fp_trend:
        analysis_parts.append(f"- **Trend:** Performance is {fp_trend} over recent games\n")

    # Efficiency note
    if avg_fg_pct > season_avg_fg_pct * 1.05 and season_avg_fg_pct > 0.40:
        analysis_parts.append(
            f"- **Efficiency:** Shooting well above season average ({avg_fg_pct:.1%} vs {season_avg_fg_pct:.1%} FG%)\n"
        )
    elif avg_fg_pct < season_avg_fg_pct * 0.95 and season_avg_fg_pct > 0.40:
        analysis_parts.append(
            f"- **Efficiency:** Shooting below season average ({avg_fg_pct:.1%} vs {season_avg_fg_pct:.1%} FG%)\n"
        )

    return "".join(analysis_parts)


def load_data_from_github_artifact(
    repo_owner="chriztopherton",
    repo_name="nbafantasizer",
    workflow_name="Update NBA Player Statistics Data",
    artifact_name="player-statistics-post-21",
    github_token=None,
    fallback_path="data/PlayerStatistics_transformed_post_21.csv",
):
    """
    Load CSV data from the latest GitHub Actions artifact.

    Downloads the latest artifact from a GitHub Actions workflow run and returns
    it as a pandas DataFrame. Falls back to a local file if the download fails.

    Args:
        repo_owner (str): GitHub repository owner (default: "chriztopherton").
        repo_name (str): GitHub repository name (default: "nbafantasizer").
        workflow_name (str): Name of the workflow (default: "Update NBA Player Statistics Data").
        artifact_name (str): Name of the artifact to download (default: "player-statistics-post-21").
        github_token (str, optional): GitHub personal access token for authentication.
                                      If None, uses GITHUB_TOKEN env var or unauthenticated requests.
        fallback_path (str): Local file path to use if artifact download fails
                            (default: "data/PlayerStatistics_transformed_post_21.csv").

    Returns:
        pd.DataFrame: The loaded CSV data as a pandas DataFrame.
    """
    # Get GitHub token from environment if not provided
    if github_token is None:
        github_token = os.getenv("GITHUB_TOKEN")

    api_base = "https://api.github.com"
    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    try:
        # Get workflow runs - try to find workflow by filename first (more reliable)
        # The workflow file is: .github/workflows/update_nba_data.yml
        # So we can search for workflows with that path
        workflow_file = "update_nba_data.yml"
        workflows_url = f"{api_base}/repos/{repo_owner}/{repo_name}/actions/workflows"
        response = requests.get(workflows_url, headers=headers, timeout=10)

        # Handle rate limiting or authentication issues
        if response.status_code == 403:
            raise ValueError("GitHub API rate limit exceeded or authentication required")
        if response.status_code == 404:
            raise ValueError(f"Repository {repo_owner}/{repo_name} not found or not accessible")

        response.raise_for_status()
        workflows_data = response.json()
        workflows = workflows_data.get("workflows", [])

        # Find the workflow by name or path
        workflow_id = None
        for workflow in workflows:
            if workflow.get("name") == workflow_name or workflow_file in workflow.get("path", ""):
                workflow_id = workflow.get("id")
                break

        if not workflow_id:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        # Get the latest successful workflow run
        runs_url = f"{api_base}/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/runs"
        response = requests.get(
            runs_url, headers=headers, params={"per_page": 1, "status": "success"}, timeout=10
        )
        response.raise_for_status()
        runs_data = response.json()
        runs = runs_data.get("workflow_runs", [])

        if not runs:
            raise ValueError("No successful workflow runs found")

        run_id = runs[0].get("id")

        # Get artifacts for this run
        artifacts_url = f"{api_base}/repos/{repo_owner}/{repo_name}/actions/runs/{run_id}/artifacts"
        response = requests.get(artifacts_url, headers=headers, timeout=10)
        response.raise_for_status()
        artifacts_data = response.json()
        artifacts = artifacts_data.get("artifacts", [])

        # Find the artifact by name
        artifact = None
        for art in artifacts:
            if art.get("name") == artifact_name:
                artifact = art
                break

        if not artifact:
            raise ValueError(f"Artifact '{artifact_name}' not found in latest run")

        # Download the artifact (it's a zip file)
        # The archive_download_url requires authentication
        download_url = artifact.get("archive_download_url")
        if not download_url:
            raise ValueError("Artifact download URL not found")

        if not github_token:
            raise ValueError("GitHub token required to download artifacts")

        response = requests.get(download_url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        # Extract CSV from zip
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            # Find the CSV file in the zip (artifact contains: data/PlayerStatistics_transformed_post_21.csv)
            csv_files = [f for f in zip_file.namelist() if f.endswith(".csv")]
            if not csv_files:
                raise ValueError("No CSV file found in artifact")

            # Read the CSV file (prefer the expected filename)
            csv_file = None
            for f in csv_files:
                if "PlayerStatistics_transformed_post_21.csv" in f:
                    csv_file = f
                    break
            if not csv_file:
                csv_file = csv_files[0]  # Use first CSV found

            csv_content = zip_file.read(csv_file)
            df = pd.read_csv(BytesIO(csv_content))

        return df

    except Exception as e:
        # Fallback to local file
        error_msg = str(e)
        print(
            f"Warning: Failed to load data from GitHub artifact ({error_msg}). Using local file: {fallback_path}"
        )
        if os.path.exists(fallback_path):
            return pd.read_csv(fallback_path)
        else:
            raise FileNotFoundError(
                f"Could not load data from GitHub artifact and local file not found: {fallback_path}"
            ) from e
