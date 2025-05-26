"""
Data Processing Script for Regime Detection Project

This script processes the raw data collected for the regime detection project,
handles missing data, and prepares it for analysis.

"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define project paths
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'data/raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data/processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Define time period (should match collect_data.py)
START_DATE = '2010-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

def load_raw_data():
    """
    Load all available raw asset data from CSV files in the RAW_DATA_DIR.

    Returns:
        dict: Dictionary of DataFrames containing raw asset data (key: asset name, value: DataFrame).
    """
    asset_data = {}
    logger.info(f"Loading raw data from: {RAW_DATA_DIR}")

    try:
        # Get all CSV files in raw data directory
        csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('_data.csv')]
        logger.info(f"Found CSV files: {csv_files}")

        if not csv_files:
            logger.warning(f"No CSV files found in {RAW_DATA_DIR}. Run collect_data.py first.")
            return asset_data

        for csv_file in csv_files:
            # Extract asset name from filename (e.g., finance_data.csv -> Finance)
            asset_name = csv_file.replace('_data.csv', '').replace('_', ' ').title()
            file_path = os.path.join(RAW_DATA_DIR, csv_file)
            try:
                # Load data, ensuring 'date' column is parsed correctly
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                # Ensure column names are consistent (lowercase)
                df.columns = [col.lower() for col in df.columns]
                asset_data[asset_name] = df
                logger.info(f"Loaded {len(df)} rows of data for {asset_name} from {csv_file}")
            except Exception as e:
                logger.error(f"Error loading file {csv_file}: {e}")

    except FileNotFoundError:
        logger.error(f"Raw data directory not found: {RAW_DATA_DIR}")
    except Exception as e:
        logger.error(f"An error occurred while loading raw data: {e}")

    return asset_data

def process_asset_data(asset_data):
    """
    Process asset data by aligning dates, handling missing values, selecting relevant columns,
    and calculating returns.

    Args:
        asset_data (dict): Dictionary of DataFrames containing raw asset data.

    Returns:
        dict: Dictionary containing processed DataFrames ('prices', 'returns').
              Returns empty dict if input is empty or processing fails.
    """
    if not asset_data:
        logger.warning("No asset data provided for processing.")
        return {}

    logger.info("Processing asset data...")

    # Extract adjusted close prices for each asset
    # Use 'adj_close' if available (from yfinance with auto_adjust=False), otherwise 'close'
    adj_close_data = {}
    for asset_name, df in asset_data.items():
        if 'adj_close' in df.columns:
            adj_close_data[asset_name] = df['adj_close']
        elif 'close' in df.columns:
            logger.warning(f"'adj_close' not found for {asset_name}, using 'close'.")
            adj_close_data[asset_name] = df['close']
        else:
            logger.warning(f"Neither 'adj_close' nor 'close' found for {asset_name}. Skipping.")

    if not adj_close_data:
        logger.error("No valid price columns found in the loaded data.")
        return {}

    # Combine into a single DataFrame
    combined_df = pd.DataFrame(adj_close_data)
    logger.info(f"Combined DataFrame shape before processing: {combined_df.shape}")

    # Align dates and handle missing values
    # Reindex to ensure a common date range (optional, depends on strategy needs)
    # combined_df = combined_df.reindex(pd.date_range(start=combined_df.index.min(), end=combined_df.index.max(), freq='B'))

    # Forward fill missing values (common practice)
    combined_df_filled = combined_df.fillna(method='ffill')

    # Optional: Backward fill remaining NaNs at the beginning
    combined_df_filled = combined_df_filled.fillna(method='bfill')

    # Drop assets/columns that are still all NaN after filling (if any)
    combined_df_filled.dropna(axis=1, how='all', inplace=True)
    # Drop rows that are still all NaN (if any, unlikely after ffill/bfill)
    combined_df_filled.dropna(axis=0, how='all', inplace=True)

    logger.info(f"Combined DataFrame shape after filling NaNs: {combined_df_filled.shape}")

    if combined_df_filled.empty:
        logger.error("DataFrame is empty after handling missing values.")
        return {}

    # Calculate daily returns
    returns_df = combined_df_filled.pct_change()

    # Remove the first row of returns (NaN)
    returns_df = returns_df.iloc[1:]

    logger.info(f"Calculated returns DataFrame shape: {returns_df.shape}")

    # Create a dictionary with processed data
    processed_data = {
        'asset_prices': combined_df_filled,
        'asset_returns': returns_df
        # Add other processed data like correlations if needed by regime_detection.py
    }

    # Example: Calculate correlation matrix (if needed by regime_detection.py)
    if 'asset_returns' in processed_data and not processed_data['asset_returns'].empty:
        try:
            corr_matrix = processed_data['asset_returns'].corr()
            processed_data['asset_correlations'] = corr_matrix
            logger.info("Calculated correlation matrix.")
        except Exception as e:
             logger.error(f"Could not calculate correlation matrix: {e}")

    # Example: Create asset class indices (if needed by regime_detection.py)
    # This part might require specific logic based on how indices were defined/used
    # For now, let's just pass the benchmark prices as a placeholder if they exist
    benchmark_cols = [col for col in processed_data['asset_prices'].columns if 'Benchmark' in col]
    if benchmark_cols:
        processed_data['asset_class_indices'] = processed_data['asset_prices'][benchmark_cols]
        logger.info("Extracted benchmark columns as placeholder for asset_class_indices.")
    else:
        logger.warning("No benchmark columns found for asset_class_indices.")
        # Create an empty DataFrame or handle as needed by regime_detection.py
        processed_data['asset_class_indices'] = pd.DataFrame(index=processed_data['asset_prices'].index)


    return processed_data

def save_processed_data(processed_data):
    """
    Save processed data to CSV files in the PROCESSED_DATA_DIR.

    Args:
        processed_data (dict): Dictionary of DataFrames containing processed data.
    """
    if not processed_data:
        logger.warning("No processed data to save.")
        return

    logger.info(f"Saving processed data to: {PROCESSED_DATA_DIR}")

    for data_type, df in processed_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            output_file = os.path.join(PROCESSED_DATA_DIR, f"{data_type}.csv")
            try:
                df.to_csv(output_file)
                logger.info(f"Saved {data_type} data ({df.shape}) to {output_file}")
            except Exception as e:
                logger.error(f"Error saving {data_type} data to {output_file}: {e}")
        else:
            logger.warning(f"Skipping save for {data_type} as it's empty or not a DataFrame.")

def main():
    """
    Main function to load raw data, process it, and save the results.
    """
    logger.info("Starting data processing script")

    # Load raw data collected by collect_data.py
    raw_asset_data = load_raw_data()

    if not raw_asset_data:
        logger.error("Failed to load raw data. Exiting.")
        return

    # Process the loaded raw data
    processed_data = process_asset_data(raw_asset_data)

    # Save the processed data
    save_processed_data(processed_data)

    logger.info("Data processing script completed")

if __name__ == "__main__":
    main()

