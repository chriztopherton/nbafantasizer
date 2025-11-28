import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import kagglehub
from kagglehub import KaggleDatasetAdapter

st.set_page_config(layout="wide")

st.title("Fantasy Points Analyzer")
st.write("This is a tool to analyze fantasy points for a player over time.")

#load data with caching to avoid redownloading on every filter change
@st.cache_data
def load_and_process_data():
    """Load and process the Kaggle dataset. This function is cached to avoid redownloading."""
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "eoinamoore/historical-nba-data-and-player-box-scores",
        "PlayerStatistics.csv",
    )
    df_fp = df.copy()
    df_fp['player_name'] = df_fp['firstName'] + ' ' + df_fp['lastName']
    if 'gameDateTimeEst' in df_fp.columns:
        df_fp['gameDateTimeEst_raw'] = df_fp['gameDateTimeEst'].copy()
        df_fp['gameDateTimeEst'] = pd.to_datetime(df_fp['gameDateTimeEst'], format='ISO8601', errors='coerce', utc=True)
        if df_fp['gameDateTimeEst'].isna().sum() > len(df_fp) * 0.1:
            mask = df_fp['gameDateTimeEst'].isna()
            df_fp.loc[mask, 'gameDateTimeEst'] = pd.to_datetime(
                df_fp.loc[mask, 'gameDateTimeEst_raw'], 
                errors='coerce',
                utc=True
            )
        # Convert to timezone-naive if needed
        if df_fp['gameDateTimeEst'].dt.tz is not None:
            df_fp['gameDateTimeEst'] = df_fp['gameDateTimeEst'].dt.tz_localize(None)
    
    return df_fp

df_fp = load_and_process_data()

# df_fp['efficiency'] = df_fp['fieldGoalsMade']/df_fp['fieldGoalsAttempted']
#calculate fantasy points
df_fp['FP'] = (
    df_fp['points'] * 1.0 +                    # Points scored
    df_fp['reboundsTotal'] * 1.2 +             # Total rebounds
    df_fp['assists'] * 1.5 +                   # Assists
    df_fp['blocks'] * 3.0 +                    # Blocked shots
    df_fp['steals'] * 3.0 +                    # Steals
    df_fp['turnovers'] * -1.0                  # Turnovers (negative)
)

# Add '@' prefix to opponent team name when playing away (home == 0)
opponent_with_at = df_fp['opponentteamName'].copy()
if 'home' in df_fp.columns:
    opponent_with_at = np.where(
        df_fp['home'] == 0,
        '@' + df_fp['opponentteamName'],
        df_fp['opponentteamName']
    )
df_fp['game_loc_date'] = df_fp['gameDateTimeEst'].astype(str) + ' ' + opponent_with_at

#display data
# st.dataframe(df_fp)

#display fantasy points over time
# st.line_chart(df_fp['FP'])

#display fantasy points over time for a specific player
# player_first_name = st.selectbox("Select a player", df_fp['firstName'].unique())
# player_last_name = st.selectbox("Select a player", df_fp['lastName'].unique())

# player_name = f"{player_first_name} {player_last_name}"
with st.sidebar:
    player_name = st.selectbox("Select a player", df_fp['player_name'].unique())

    # Add date range filters
    col1, col2 = st.columns(2)
    with col1:
        start_date_filter = st.date_input(
            "Start Date",
            value=datetime(2024, 10, 22).date(),
            min_value=datetime(2015, 1, 1).date(),
            max_value=datetime.now().date()
        )
    with col2:
        end_date_filter = st.date_input(
            "End Date",
            value=datetime.now().date(),
            min_value=datetime(2015, 1, 1).date(),
            max_value=datetime.now().date()
        )

# Validate date range
if start_date_filter > end_date_filter:
    st.error("⚠️ Start date must be before or equal to end date. Please adjust your date selection.")
    st.stop()

#display fantasy points over time for a specific player
player_data = df_fp[df_fp['player_name'] == player_name].copy()

# Find and convert date column
date_col = None
for col in ['gameDateTimeEst', 'game_date', 'date', 'GAME_DATE', 'DATE_EST']:
    if col in player_data.columns:
        date_col = col
        break

if date_col:
    if len(player_data) > 0:
        if not pd.api.types.is_datetime64_any_dtype(player_data[date_col]):
            player_data[date_col] = pd.to_datetime(player_data[date_col], errors='coerce', utc=True)
            if player_data[date_col].dt.tz is not None:
                player_data[date_col] = player_data[date_col].dt.tz_localize(None)
        else:
            if player_data[date_col].dt.tz is not None:
                player_data[date_col] = player_data[date_col].dt.tz_localize(None)
            
            nan_count = player_data[date_col].isna().sum()
            if nan_count > 0:
                if 'gameDateTimeEst_raw' in player_data.columns:
                    mask = player_data[date_col].isna()
                    if mask.sum() > 0:
                        player_data.loc[mask, date_col] = pd.to_datetime(
                            player_data.loc[mask, 'gameDateTimeEst_raw'], 
                            errors='coerce', 
                            format='ISO8601'
                        )
                        if player_data[date_col].dt.tz is not None:
                            player_data[date_col] = player_data[date_col].dt.tz_localize(None)
    
    player_data = player_data.dropna(subset=[date_col, 'FP']).copy()
    
    start_date = pd.Timestamp(start_date_filter).normalize()
    end_date = pd.Timestamp(end_date_filter).normalize()
    player_data['date_only'] = pd.to_datetime(player_data[date_col]).dt.normalize()
    
    player_data = player_data[
        (player_data['date_only'] >= start_date) & 
        (player_data['date_only'] <= end_date)
    ].copy()
    player_data = player_data.sort_values(date_col)
    
    if len(player_data) > 0:
        # player_data is already filtered to the selected date range, so use it directly
        player_data_ytd = player_data.copy()
        
        if len(player_data_ytd) > 0:
            # Calculate aggregated averages for quantitative metrics over different time windows
            # Get the most recent date
            most_recent_date = player_data_ytd[date_col].max()
            
            # Define time windows (in days)
            time_windows = {
                'Last 7 days (avg)': 7,
                'Last 14 days (avg)': 14,
                'Last 30 days (avg)': 30,
                'Last 90 days (avg)': 90,
                'YTD (avg)': None  # None means use all YTD data
            }
            
            # Build aggregated data for all time windows
            aggregated_rows = []
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
                        'Time Window': window_name,
                        'Fantasy Points': window_data['FP'].mean(),
                        'Points': window_data['points'].mean(),
                        'Rebounds': window_data['reboundsTotal'].mean(),
                        'Assists': window_data['assists'].mean(),
                        'Blocks': window_data['blocks'].mean(),
                        'Steals': window_data['steals'].mean(),
                        'Turnovers': window_data['turnovers'].mean()
                    }
                    aggregated_rows.append(row)
            
            # Create single DataFrame with all aggregated averages
            aggregated_df = pd.DataFrame(aggregated_rows)
            
            game_log, stats = st.tabs(["Game Log", "Stats"])
            with game_log:
                st.subheader("Game Log")
                # Prepare dataframe for display
                display_df = player_data_ytd[['game_loc_date','FP','numMinutes','points','reboundsTotal','assists','steals','blocks','turnovers','fieldGoalsPercentage','freeThrowsPercentage','plusMinusPoints']].sort_values(by='game_loc_date', ascending=False).copy()
                
                # Format numeric columns as integers (except FP, fieldGoalsPercentage, freeThrowsPercentage)
                integer_columns = ['numMinutes', 'points', 'reboundsTotal', 'assists', 'steals', 'blocks', 'turnovers', 'plusMinusPoints']
                for col in integer_columns:
                    if col in display_df.columns:
                        # Round and convert to nullable integer type
                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round().astype('Int64')
                
                # Format decimal columns to 2 decimal places
                decimal_columns = ['FP', 'fieldGoalsPercentage', 'freeThrowsPercentage']
                for col in decimal_columns:
                    if col in display_df.columns:
                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(2)
                
                # Apply green gradient styling to specified columns
                columns_to_style = ['FP','numMinutes','points', 'reboundsTotal', 'assists', 'steals', 'blocks', 'fieldGoalsPercentage', 'freeThrowsPercentage']
                
                # Only style columns that exist in the dataframe
                columns_to_style = [col for col in columns_to_style if col in display_df.columns]
                
                # Create styled dataframe with green gradient
                # Format decimal columns to always show 2 decimal places
                format_dict = {}
                for col in decimal_columns:
                    if col in display_df.columns:
                        format_dict[col] = '{:.2f}'
                
                styled_df = display_df.style.format(format_dict).background_gradient(
                    subset=columns_to_style,
                    cmap='Greens',
                    axis=0  # Apply gradient along rows (column-wise)
                ).background_gradient(
                    subset='turnovers',
                    cmap='Reds',
                    axis=0  # Apply gradient along rows (column-wise)
                )
                
                st.dataframe(styled_df, width='stretch')
                # st.dataframe(player_data_ytd, use_container_width=True)
            with stats:
                st.subheader("Stats")
                # st.dataframe(player_data_ytd, use_container_width=True)
            
                # Display aggregated averages as a single table
                st.subheader("Aggregated Averages by Time Window")
                if len(aggregated_df) > 0:
                    st.dataframe(aggregated_df, width='stretch')
            
            # Use YTD data for the main chart
            x = player_data_ytd[date_col].values
            y = player_data_ytd['FP'].values
            
            # Calculate moving averages for different time windows
            player_data_indexed = player_data_ytd.set_index(date_col).copy()
            
            # Define moving average windows and colors
            ma_windows = {
                '7 days': ('7D', '#FF6B6B'),   # Red
                '14 days': ('14D', '#4ECDC4'),  # Teal
                '30 days': ('30D', '#45B7D1'),  # Blue
                '90 days': ('90D', '#FFA07A')   # Light salmon
            }
            
            # Calculate moving averages
            moving_averages = {}
            for ma_name, (window_size, color) in ma_windows.items():
                ma_series = player_data_indexed['FP'].rolling(window=window_size, center=True, min_periods=1).mean()
                moving_averages[ma_name] = {
                    'values': ma_series.values,
                    'color': color
                }
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add actual data line (green like Robinhood stock prices)
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers+text',
                name='Fantasy Points',
                line=dict(color='#00C805', width=0.5),  # Robinhood green
                marker=dict(color='#00C805', size=4, opacity=0.6),
                text=[f'{val:.1f}' for val in y],
                textposition='top center',
                textfont=dict(size=9, color='#00C805')
            ))
            
            # Add moving average smoothing lines
            for ma_name, ma_data in moving_averages.items():
                fig.add_trace(go.Scatter(
                    x=x,
                    y=ma_data['values'],
                    mode='lines',
                    name=f'MA ({ma_name})',
                    line=dict(color=ma_data['color'], width=2),
                    opacity=0.8
                ))
            
            # Update layout with dark background
            start_date_str = start_date_filter.strftime('%b %d, %Y')
            end_date_str = end_date_filter.strftime('%b %d, %Y')
            fig.update_layout(
                title=dict(
                    text=f'Fantasy Points Over Time: {player_name} ({start_date_str} - {end_date_str})',
                    font=dict(size=20, color='white')
                ),
                xaxis=dict(
                    title='Date',
                    showgrid=True,
                    gridcolor='#333333',
                    gridwidth=1,
                    showline=True,
                    linecolor='#555555',
                    title_font=dict(size=14, color='#cccccc'),
                    tickfont=dict(color='#cccccc')
                ),
                yaxis=dict(
                    title='Fantasy Points',
                    showgrid=True,
                    gridcolor='#333333',
                    gridwidth=1,
                    showline=True,
                    linecolor='#555555',
                    title_font=dict(size=14, color='#cccccc'),
                    tickfont=dict(color='#cccccc')
                ),
                hovermode='x unified',
                plot_bgcolor='black',
                paper_bgcolor='black',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(0, 0, 0, 0.8)',
                    bordercolor='#555555',
                    borderwidth=1,
                    font=dict(size=12, color='white')
                ),
                margin=dict(l=60, r=20, t=60, b=50)
            )
            
            st.plotly_chart(fig, width='stretch')
            
    
        else:
            st.write(f"No data available for {player_name} in the selected date range ({start_date_filter} to {end_date_filter}).")
    else:
        st.write(f"No data available for {player_name} in the selected date range ({start_date_filter} to {end_date_filter}).")
else:
    st.write("No date column found in the dataset. Cannot display time-based chart.")