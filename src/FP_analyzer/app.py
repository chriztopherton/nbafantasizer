from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from injury_scraper import ESPNInjuryScraper
from player_analysis import render_player_analysis_tab
from style import apply_custom_css
from trade_analyzer import render_trade_analyzer_tab
from utils import get_player_attributes, get_player_image_url, get_player_person_id

# Load environment variables
load_dotenv()

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

        # Display player attributes
        player_attrs = get_player_attributes(player_name, df_fp, injury_scraper)
        if player_attrs:
            attrs_parts = []
            if player_attrs.get("team"):
                attrs_parts.append(f"**Team:** {player_attrs['team']}")
            if player_attrs.get("position"):
                attrs_parts.append(f"**Position:** {player_attrs['position']}")
            if player_attrs.get("height"):
                attrs_parts.append(f"**Height:** {player_attrs['height']}")

            if attrs_parts:
                for attr_line in attrs_parts:
                    st.markdown(attr_line)

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
    render_player_analysis_tab(
        player_name, df_fp, injury_scraper, start_date_filter, end_date_filter
    )

# Tab 2: Trade Analyzer
with tab2:
    render_trade_analyzer_tab(
        df_fp, player_list, injury_scraper, start_date_filter, end_date_filter
    )
