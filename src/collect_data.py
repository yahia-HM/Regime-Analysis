"""
Data Collection Script for Regime Detection Project

This script collects historical price data for European sector ETFs and indices using yfinance.
The data is saved to CSV files in the data/raw directory.

"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define project paths
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'data/raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Define European sector ETFs and indices relevant to regime detection
# (Adjust these based on the actual needs of regime_detection.py if different)
ASSET_TICKERS = {
    'Finance': 'EUFN',       # iShares MSCI Europe Financials ETF
    'Technology': 'FEZ',     # SPDR EURO STOXX 50 ETF
    'Healthcare': 'HEAL.L',  # iShares Healthcare Innovation UCITS ETF
    'Energy': 'IESU.L',      # iShares STOXX Europe 600 Utilities
    'Consumer': 'EXH7.DE',   # iShares STOXX Europe 600 Personal & Household Goods UCITS ETF
    'Industrial': 'EXH4.DE', # iShares STOXX Europe 600 Industrial Goods & Services UCITS ETF
    'Materials': 'EXV6.DE',  # iShares STOXX Europe 600 Basic Resources UCITS ETF
    'Benchmark_STOXX600': '^STOXX',     # STOXX Europe 600 Index
    'Benchmark_EUROSTOXX50': '^STOXX50E' # EURO STOXX 50 Index
}
# Define time period for data collection
START_DATE = '2010-01-01' # Extended start date for potentially longer history needed by HMM
END_DATE = datetime.now().strftime('%Y-%m-%d')

def download_asset_data(symbol, start_date, end_date):
    """
    Download historical price data for a given symbol using yfinance.

    Args:
        symbol (str): The ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pandas.DataFrame: DataFrame containing the historical price data, or None if download fails.
    """
    logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
    try:
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
        
        if data.empty:
            logger.warning(f"No data downloaded for {symbol}. It might be an invalid ticker or delisted for the period.")
            return None
        # Rename 'Adj Close' to 'adj_close' for consistency with original script if needed by process_data.py
        
        data.columns = data.columns.droplevel(1)
        if 'Adj Close' in data.columns:
             data.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
        # Ensure column names are lowercase for consistency
        data.columns = [col.lower() for col in data.columns]

        logger.info(f"Successfully downloaded {len(data)} rows of data for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {str(e)}")
        return None

def main():
    """
    Main function to download data for all assets and save to CSV files.
    """
    logger.info("Starting data collection process using yfinance")

    # Download data for each asset
    for asset_name, symbol in ASSET_TICKERS.items():
        logger.info(f"Processing {asset_name} (Symbol: {symbol})")

        # Download data
        df = download_asset_data(symbol, START_DATE, END_DATE)

        if df is not None and not df.empty:
            # Save to CSV
            output_file = os.path.join(RAW_DATA_DIR, f"{asset_name.lower()}_data.csv")
            df.to_csv(output_file)
            logger.info(f"Saved data to {output_file}")
        else:
            logger.warning(f"No data available or error for {asset_name} (Symbol: {symbol})")

    logger.info("Data collection process completed")

if __name__ == "__main__":
    main()
