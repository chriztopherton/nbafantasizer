# Streamlit Player Research Tool

A web-based interface for the Fantasy Player Research Tool.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Features

### 📊 Overview Tab
- View league information
- Browse all players in the league
- Search and filter players by name and position
- See player status and injury information

### 🏥 Injury Report Tab
- View all injured players in the league
- See injury status and notes
- Identify which injured players are currently rostered

### 💎 Waiver Wire Gems Tab
- Find potential waiver wire pickups
- Filter by ownership percentage and position
- Identify injured players who might be dropped

### ⚖️ Player Comparison Tab
- Compare statistics for multiple players
- Select season or specific week stats
- Side-by-side comparison view

### 📈 Trend Analysis Tab
- Analyze player performance trends across multiple weeks
- Week-by-week performance breakdown
- Identify improving or declining players

## Usage

1. **Load League**: Enter your league ID (e.g., 82771) or full league key (e.g., 466.l.82771) in the sidebar
2. **Collect Players**: Click "Collect Players" to gather all player data from the league
3. **Explore**: Use the tabs to explore different features

## Notes

- The app uses session state to cache data, so you don't need to reload the league on every interaction
- Player collection may take a few moments depending on league size
- Some features require player data to be collected first
