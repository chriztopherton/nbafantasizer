#!/usr/bin/env python3
"""
Script to export NBA player statistics data post-2021-10-19 to CSV.

This script loads data from Kaggle, processes it, filters for dates >= 2021-10-19,
and exports to PlayerStatistics_transformed_post_21.csv
"""

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np

# Set pandas display options
pd.set_option('display.max_columns', None)


def load_and_process_data():
    """Load and process the Kaggle dataset."""
    try:
        # Load with low_memory=False to avoid dtype warnings and improve performance
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "eoinamoore/historical-nba-data-and-player-box-scores",
            "PlayerStatistics.csv",
        )
    except Exception as e:
        print(f"Error loading dataset from Kaggle: {e}")
        raise
    
    # Process in-place where possible to save memory
    df["player_name"] = df["firstName"] + " " + df["lastName"]
    
    if "gameDateTimeEst" in df.columns:
        # Store raw copy only if needed for fallback
        df["gameDateTimeEst"] = pd.to_datetime(
            df["gameDateTimeEst"], format="ISO8601", errors="coerce", utc=True
        )
        if df["gameDateTimeEst"].isna().sum() > len(df) * 0.1:
            # Only create raw copy if we need it for fallback
            if "gameDateTimeEst_raw" not in df.columns:
                df["gameDateTimeEst_raw"] = df["gameDateTimeEst"].copy()
            mask = df["gameDateTimeEst"].isna()
            df.loc[mask, "gameDateTimeEst"] = pd.to_datetime(
                df.loc[mask, "gameDateTimeEst_raw"], errors="coerce", utc=True
            )
        # Convert to timezone-naive if needed
        if df["gameDateTimeEst"].dt.tz is not None:
            df["gameDateTimeEst"] = df["gameDateTimeEst"].dt.tz_localize(None)
    
    # Calculate fantasy points
    df["FP"] = (
        df["points"] * 1.0  # Points scored
        + df["reboundsTotal"] * 1.2  # Total rebounds
        + df["assists"] * 1.5  # Assists
        + df["blocks"] * 3.0  # Blocked shots
        + df["steals"] * 3.0  # Steals
        + df["turnovers"] * -1.0  # Turnovers (negative)
    )
    
    # Add '@' prefix to opponent team name when playing away (home == 0)
    # Use vectorized operation instead of copying
    if "home" in df.columns:
        df["opponent_with_at"] = np.where(
            df["home"] == 0, "@" + df["opponentteamName"], df["opponentteamName"]
        )
    else:
        df["opponent_with_at"] = df["opponentteamName"]
    
    # Create game_loc_date more efficiently
    df["game_loc_date"] = df["gameDateTimeEst"].astype(str) + " " + df["opponent_with_at"]

    # Return a copy to ensure the cached data is immutable
    df_copy = df.copy()
    
    return df_copy


def main():
    """Main function to load, process, filter, and export data."""
    print("Loading and processing data from Kaggle...")
    df_fp = load_and_process_data()
    
    print(f"Total records loaded: {len(df_fp)}")
    
    # Filter for dates >= 2021-10-19
    print("Filtering for dates >= 2021-10-19...")
    post_21 = df_fp[df_fp['gameDateTimeEst'] >= '2021-10-19']
    
    print(f"Records after filtering: {len(post_21)}")
    
    # Export to CSV
    output_file = 'data/PlayerStatistics_transformed_post_21.csv'
    print(f"Exporting to {output_file}...")
    post_21.to_csv(output_file, index=False)
    
    print(f"✅ Successfully exported {len(post_21)} records to {output_file}")


if __name__ == "__main__":
    main()

