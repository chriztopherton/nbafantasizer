"""Player profile header and UI components."""

import pandas as pd
import streamlit as st
from utils import get_player_image_url, get_player_person_id


def calculate_player_rank(player_name, df_fp):
    """
    Calculate player's rank based on average fantasy points.

    Args:
        player_name (str): Name of the player.
        df_fp (pd.DataFrame): DataFrame containing all player data.

    Returns:
        int or None: Player rank if calculable, None otherwise.
    """
    if player_name is None or df_fp is None or len(df_fp) == 0:
        return None

    try:
        # Calculate average FP for all players
        player_avg_fp = df_fp[df_fp["player_name"] == player_name]["FP"].mean()
        if pd.isna(player_avg_fp):
            return None

        # Calculate average FP for each player and rank
        player_avg_fps = df_fp.groupby("player_name")["FP"].mean().sort_values(ascending=False)
        rank = (player_avg_fps > player_avg_fp).sum() + 1
        return rank
    except Exception:
        return None


def render_player_profile_header(player_name, df_fp, player_id=None):
    """
    Render player profile header similar to Yahoo Fantasy UI.

    Args:
        player_name (str): Name of the player.
        df_fp (pd.DataFrame): DataFrame containing player data.
        player_id (int, optional): NBA player ID.
    """
    if player_name is None:
        return

    # Get player ID if not provided
    if player_id is None:
        player_id = get_player_person_id(player_name, df_fp)

    # Get player image
    image_url = get_player_image_url(player_id) if player_id else None

    # Get player attributes
    from utils import get_player_attributes

    # Try to get injury scraper if available, otherwise pass None
    try:
        from injury_scraper import ESPNInjuryScraper

        injury_scraper = ESPNInjuryScraper()
    except Exception:
        injury_scraper = None

    player_attrs = get_player_attributes(player_name, df_fp, injury_scraper)

    # Calculate season stats
    player_data = df_fp[df_fp["player_name"] == player_name]
    if len(player_data) == 0:
        return

    # Calculate averages
    gp = len(player_data)
    avg_pts = player_data["points"].mean() if "points" in player_data.columns else 0
    avg_reb = player_data["reboundsTotal"].mean() if "reboundsTotal" in player_data.columns else 0
    avg_ast = player_data["assists"].mean() if "assists" in player_data.columns else 0
    avg_stl = player_data["steals"].mean() if "steals" in player_data.columns else 0
    avg_blk = player_data["blocks"].mean() if "blocks" in player_data.columns else 0

    # Calculate rank
    rank = calculate_player_rank(player_name, df_fp)

    # Get team and position
    team = player_attrs.get("team", "Unknown") if player_attrs else "Unknown"
    position = player_attrs.get("position", "N/A") if player_attrs else "N/A"

    # Get player number if available (from most recent game)
    player_number = None
    if "playerJerseyNumber" in player_data.columns:
        player_number = (
            player_data["playerJerseyNumber"].dropna().iloc[-1] if len(player_data) > 0 else None
        )

    # Render header with custom CSS
    st.markdown(
        """
        <style>
        .player-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: white;
        }
        .player-header-content {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .player-image {
            border-radius: 50%;
            border: 3px solid white;
        }
        .player-info {
            flex: 1;
        }
        .player-name {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .player-details {
            font-size: 16px;
            margin-bottom: 15px;
            opacity: 0.9;
        }
        .player-stats-row {
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 20px;
            font-weight: bold;
        }
        .stat-label {
            font-size: 12px;
            opacity: 0.8;
            margin-top: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Build stats HTML
    stats_html = ""
    if rank:
        stats_html += f'<div class="stat-item"><div class="stat-value">{rank}</div><div class="stat-label">Rank</div></div>'
    stats_html += f'<div class="stat-item"><div class="stat-value">{gp}</div><div class="stat-label">GP</div></div>'
    stats_html += f'<div class="stat-item"><div class="stat-value">{avg_pts:.1f}</div><div class="stat-label">PTS</div></div>'
    stats_html += f'<div class="stat-item"><div class="stat-value">{avg_reb:.1f}</div><div class="stat-label">REB</div></div>'
    stats_html += f'<div class="stat-item"><div class="stat-value">{avg_ast:.1f}</div><div class="stat-label">AST</div></div>'
    stats_html += f'<div class="stat-item"><div class="stat-value">{avg_stl:.1f}</div><div class="stat-label">ST</div></div>'
    stats_html += f'<div class="stat-item"><div class="stat-value">{avg_blk:.1f}</div><div class="stat-label">BLK</div></div>'

    # Build player details text
    details_text = position if position else ""
    if team and team != "Unknown":
        if details_text:
            details_text += f" • {team}"
        else:
            details_text = team
    if player_number:
        if details_text:
            details_text += f" #{player_number}"
        else:
            details_text = f"#{player_number}"

    # Fallback if no details available
    if not details_text:
        details_text = "N/A"

    # Render header
    header_html = f"""
    <div class="player-header">
        <div class="player-header-content">
            {"<img src='" + image_url + "' class='player-image' width='120' height='120' />" if image_url else ""}
            <div class="player-info">
                <div class="player-name">{player_name}</div>
                <div class="player-details">{details_text}</div>
                <div class="player-stats-row">
                    {stats_html}
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
