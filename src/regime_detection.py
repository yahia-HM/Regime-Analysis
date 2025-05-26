#!/usr/bin/env python3
"""
Regime Detection Algorithm for European Cross-Asset Portfolio

This script implements a Hidden Markov Model (HMM) to detect market regimes
across different asset classes. It also implements volatility regime classification
and correlation regime analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define project paths
PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')

# Create directories if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

def load_processed_data():
    """
    Load processed data files.
    
    Returns:
    --------
    tuple
        (prices_df, returns_df, indices_df, corr_matrix) - DataFrames with processed data
    """
    prices_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'asset_prices.csv'), index_col=0, parse_dates=True)
    returns_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'asset_returns.csv'), index_col=0, parse_dates=True)
    indices_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'asset_class_indices.csv'), index_col=0, parse_dates=True)
    corr_matrix = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'asset_correlations.csv'), index_col=0)
    
    return prices_df, returns_df, indices_df, corr_matrix

def detect_market_regimes_hmm(returns_series, n_regimes=3, n_iter=1000):
    """
    Detect market regimes using Hidden Markov Model.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Series of returns
    n_regimes : int
        Number of regimes to detect
    n_iter : int
        Number of iterations for HMM training
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with regime information
    """
    # Prepare data for HMM
    X = returns_series.dropna().values.reshape(-1, 1)
    
    # Create and fit HMM
    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=n_iter, random_state=42)
    model.fit(X)
    
    # Predict hidden states
    hidden_states = model.predict(X)
    
    # Create DataFrame with regime information
    regimes = pd.DataFrame(hidden_states, index=returns_series.dropna().index, columns=['Regime'])
    
    # Calculate regime characteristics
    for i in range(n_regimes):
        mask = (regimes['Regime'] == i)
        regimes.loc[mask, 'Mean'] = returns_series.dropna()[mask].mean()
        regimes.loc[mask, 'Volatility'] = returns_series.dropna()[mask].std()
    
    # Label regimes based on characteristics
    regimes['RegimeType'] = 'Unknown'
    
    # Sort regimes by mean return
    regime_stats = regimes.groupby('Regime')[['Mean', 'Volatility']].mean().sort_values('Mean')
    regime_mapping = {
        regime_stats.index[0]: 'Bear',
        regime_stats.index[-1]: 'Bull'
    }
    
    # If we have 3 regimes, the middle one is 'Sideways'
    if n_regimes == 3:
        regime_mapping[regime_stats.index[1]] = 'Sideways'
    
    # Apply mapping
    for regime, regime_type in regime_mapping.items():
        regimes.loc[regimes['Regime'] == regime, 'RegimeType'] = regime_type
    
    return regimes

def detect_volatility_regimes(volatility_series, n_regimes=3):
    """
    Detect volatility regimes using K-means clustering.
    
    Parameters:
    -----------
    volatility_series : pd.Series
        Series of volatility values
    n_regimes : int
        Number of regimes to detect
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with volatility regime information
    """
    # Prepare data for clustering
    X = volatility_series.dropna().values.reshape(-1, 1)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_regimes, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create DataFrame with regime information
    vol_regimes = pd.DataFrame(clusters, index=volatility_series.dropna().index, columns=['VolRegime'])
    
    # Calculate regime characteristics
    for i in range(n_regimes):
        mask = (vol_regimes['VolRegime'] == i)
        vol_regimes.loc[mask, 'MeanVol'] = volatility_series.dropna()[mask].mean()
    
    # Label regimes based on characteristics
    vol_regimes['VolRegimeType'] = 'Unknown'
    
    # Sort regimes by mean volatility
    regime_stats = vol_regimes.groupby('VolRegime')['MeanVol'].mean().sort_values()
    regime_mapping = {
        regime_stats.index[0]: 'Low',
        regime_stats.index[-1]: 'High'
    }
    
    # If we have 3 regimes, the middle one is 'Medium'
    if n_regimes == 3:
        regime_mapping[regime_stats.index[1]] = 'Medium'
    
    # Apply mapping
    for regime, regime_type in regime_mapping.items():
        vol_regimes.loc[vol_regimes['VolRegime'] == regime, 'VolRegimeType'] = regime_type
    
    return vol_regimes

def detect_correlation_regimes(returns_df, window=60, n_regimes=2):
    """
    Detect correlation regimes based on average pairwise correlations.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame with asset returns
    window : int
        Rolling window size for correlation calculation
    n_regimes : int
        Number of regimes to detect
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with correlation regime information
    """
    # Filter for daily returns columns
    daily_returns = returns_df.copy()
    
    # Calculate rolling correlation matrix
    corr_matrices = {}
    for i in range(window, len(daily_returns)):
        window_data = daily_returns.iloc[i-window:i]
        corr_matrices[daily_returns.index[i]] = window_data.corr()
    
    # Calculate average pairwise correlation
    avg_correlations = pd.Series(index=list(corr_matrices.keys()))
    for date, corr_matrix in corr_matrices.items():
        # Get upper triangle of correlation matrix (excluding diagonal)
        upper_triangle = np.triu(corr_matrix.values, k=1)
        # Calculate average of upper triangle
        avg_correlations[date] = np.mean(upper_triangle[upper_triangle != 0])
    
    # Detect regimes using K-means
    X = avg_correlations.dropna().values.reshape(-1, 1)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_regimes, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create DataFrame with regime information
    corr_regimes = pd.DataFrame(clusters, index=avg_correlations.dropna().index, columns=['CorrRegime'])
    corr_regimes['AvgCorrelation'] = avg_correlations.dropna()
    
    # Label regimes based on characteristics
    corr_regimes['CorrRegimeType'] = 'Unknown'
    
    # Sort regimes by mean correlation
    regime_stats = corr_regimes.groupby('CorrRegime')['AvgCorrelation'].mean().sort_values()
    regime_mapping = {
        regime_stats.index[0]: 'Low',
        regime_stats.index[-1]: 'High'
    }
    
    # Apply mapping
    for regime, regime_type in regime_mapping.items():
        corr_regimes.loc[corr_regimes['CorrRegime'] == regime, 'CorrRegimeType'] = regime_type
    
    return corr_regimes, avg_correlations

def combine_regime_information(market_regimes, vol_regimes, corr_regimes):
    """
    Combine different regime information into a unified regime dataset.
    
    Parameters:
    -----------
    market_regimes : pd.DataFrame
        DataFrame with market regime information
    vol_regimes : pd.DataFrame
        DataFrame with volatility regime information
    corr_regimes : pd.DataFrame
        DataFrame with correlation regime information
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with combined regime information
    """
    # Create a common date range
    common_dates = sorted(set(market_regimes.index) & set(vol_regimes.index) & set(corr_regimes.index))
    
    # Create combined DataFrame
    combined_regimes = pd.DataFrame(index=common_dates)
    
    # Add market regime information
    combined_regimes['MarketRegime'] = market_regimes.loc[common_dates, 'Regime']
    combined_regimes['MarketRegimeType'] = market_regimes.loc[common_dates, 'RegimeType']
    combined_regimes['MeanReturn'] = market_regimes.loc[common_dates, 'Mean']
    combined_regimes['ReturnVolatility'] = market_regimes.loc[common_dates, 'Volatility']
    
    # Add volatility regime information
    combined_regimes['VolRegime'] = vol_regimes.loc[common_dates, 'VolRegime']
    combined_regimes['VolRegimeType'] = vol_regimes.loc[common_dates, 'VolRegimeType']
    combined_regimes['MeanVolatility'] = vol_regimes.loc[common_dates, 'MeanVol']
    
    # Add correlation regime information
    combined_regimes['CorrRegime'] = corr_regimes.loc[common_dates, 'CorrRegime']
    combined_regimes['CorrRegimeType'] = corr_regimes.loc[common_dates, 'CorrRegimeType']
    combined_regimes['AvgCorrelation'] = corr_regimes.loc[common_dates, 'AvgCorrelation']
    
    return combined_regimes

def plot_market_regimes(returns_series, market_regimes, title):
    """
    Plot market regimes with returns.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Series of returns
    market_regimes : pd.DataFrame
        DataFrame with market regime information
    title : str
        Plot title
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot returns
    ax1 = plt.subplot(2, 1, 1)
    returns_series.plot(ax=ax1, color='gray', alpha=0.7)
    ax1.set_title(f'{title} Returns')
    ax1.set_ylabel('Daily Return')
    ax1.grid(True)
    
    # Plot regimes
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    
    # Get unique regimes and their types
    regimes = market_regimes['Regime'].unique()
    regime_types = {regime: market_regimes.loc[market_regimes['Regime'] == regime, 'RegimeType'].iloc[0] 
                   for regime in regimes}
    
    # Define colors for different regime types
    colors = {'Bull': 'green', 'Bear': 'red', 'Sideways': 'blue', 'Unknown': 'gray'}
    
    # Plot each regime
    for regime in regimes:
        mask = (market_regimes['Regime'] == regime)
        regime_type = regime_types[regime]
        color = colors.get(regime_type, 'gray')
        
        # Plot regime periods
        for i in range(len(mask)):
            if mask.iloc[i]:
                ax2.axvspan(mask.index[i], mask.index[i+1] if i+1 < len(mask) else mask.index[i], 
                           alpha=0.3, color=color)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[regime_type], alpha=0.3, label=regime_type)
                      for regime_type in set(regime_types.values())]
    ax2.legend(handles=legend_elements)
    
    ax2.set_title(f'{title} Market Regimes')
    ax2.set_ylabel('Regime')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'{title.lower().replace(" ", "_")}_market_regimes.png'))
    plt.close()

def plot_volatility_regimes(volatility_series, vol_regimes, title):
    """
    Plot volatility regimes.
    
    Parameters:
    -----------
    volatility_series : pd.Series
        Series of volatility values
    vol_regimes : pd.DataFrame
        DataFrame with volatility regime information
    title : str
        Plot title
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot volatility
    ax1 = plt.subplot(2, 1, 1)
    volatility_series.plot(ax=ax1, color='gray', alpha=0.7)
    ax1.set_title(f'{title} Volatility')
    ax1.set_ylabel('Volatility')
    ax1.grid(True)
    
    # Plot regimes
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    
    # Get unique regimes and their types
    regimes = vol_regimes['VolRegime'].unique()
    regime_types = {regime: vol_regimes.loc[vol_regimes['VolRegime'] == regime, 'VolRegimeType'].iloc[0] 
                   for regime in regimes}
    
    # Define colors for different regime types
    colors = {'Low': 'green', 'Medium': 'blue', 'High': 'red', 'Unknown': 'gray'}
    
    # Plot each regime
    for regime in regimes:
        mask = (vol_regimes['VolRegime'] == regime)
        regime_type = regime_types[regime]
        color = colors.get(regime_type, 'gray')
        
        # Plot regime periods
        for i in range(len(mask)):
            if mask.iloc[i]:
                ax2.axvspan(mask.index[i], mask.index[i+1] if i+1 < len(mask) else mask.index[i], 
                           alpha=0.3, color=color)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[regime_type], alpha=0.3, label=regime_type)
                      for regime_type in set(regime_types.values())]
    ax2.legend(handles=legend_elements)
    
    ax2.set_title(f'{title} Volatility Regimes')
    ax2.set_ylabel('Regime')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'{title.lower().replace(" ", "_")}_volatility_regimes.png'))
    plt.close()

def plot_correlation_regimes(avg_correlations, corr_regimes, title):
    """
    Plot correlation regimes.
    
    Parameters:
    -----------
    avg_correlations : pd.Series
        Series of average correlations
    corr_regimes : pd.DataFrame
        DataFrame with correlation regime information
    title : str
        Plot title
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot average correlations
    ax1 = plt.subplot(2, 1, 1)
    avg_correlations.plot(ax=ax1, color='gray', alpha=0.7)
    ax1.set_title(f'{title} Average Correlations')
    ax1.set_ylabel('Average Correlation')
    ax1.grid(True)
    
    # Plot regimes
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    
    # Get unique regimes and their types
    regimes = corr_regimes['CorrRegime'].unique()
    regime_types = {regime: corr_regimes.loc[corr_regimes['CorrRegime'] == regime, 'CorrRegimeType'].iloc[0] 
                   for regime in regimes}
    
    # Define colors for different regime types
    colors = {'Low': 'green', 'High': 'red', 'Unknown': 'gray'}
    
    # Plot each regime
    for regime in regimes:
        mask = (corr_regimes['CorrRegime'] == regime)
        regime_type = regime_types[regime]
        color = colors.get(regime_type, 'gray')
        
        # Plot regime periods
        for i in range(len(mask)):
            if mask.iloc[i]:
                ax2.axvspan(mask.index[i], mask.index[i+1] if i+1 < len(mask) else mask.index[i], 
                           alpha=0.3, color=color)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[regime_type], alpha=0.3, label=regime_type)
                      for regime_type in set(regime_types.values())]
    ax2.legend(handles=legend_elements)
    
    ax2.set_title(f'{title} Correlation Regimes')
    ax2.set_ylabel('Regime')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'{title.lower().replace(" ", "_")}_correlation_regimes.png'))
    plt.close()

def plot_combined_regimes(combined_regimes, title):
    """
    Plot combined regime information.
    
    Parameters:
    -----------
    combined_regimes : pd.DataFrame
        DataFrame with combined regime information
    title : str
        Plot title
    """
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot market regimes
    ax1 = plt.subplot(3, 1, 1)
    
    # Get unique market regimes<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>
    """
    Plot combined regime information.
    
    Parameters:
    -----------
    combined_regimes : pd.DataFrame
        DataFrame with combined regime information
    title : str
        Plot title
    """
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Plot market regimes
    ax1 = plt.subplot(3, 1, 1)
    market_regime_types = combined_regimes["MarketRegimeType"].unique()
    market_colors = {"Bull": "green", "Bear": "red", "Sideways": "blue", "Unknown": "lightgray"}
    for regime_type in market_regime_types:
        mask = (combined_regimes["MarketRegimeType"] == regime_type)
        color = market_colors.get(regime_type, "lightgray")
        ax1.fill_between(combined_regimes.index, 0, 1, where=mask, color=color, alpha=0.3, transform=ax1.get_xaxis_transform())
    ax1.plot(combined_regimes.index, combined_regimes["MeanReturn"].fillna(0), color="black", linewidth=0.5, label="Mean Return (in regime)")
# Plot something on the axis
    ax1.set_title(f"{title} - Market Regimes")
    ax1.set_ylabel("Market Regime")
    ax1.set_yticks([])
    from matplotlib.patches import Patch
    legend_elements_market = [Patch(facecolor=market_colors[rt], alpha=0.3, label=rt) for rt in market_regime_types if rt in market_colors]
    ax1.legend(handles=legend_elements_market, loc="upper right")
    ax1.grid(True, axis="x")

    # Plot volatility regimes
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    vol_regime_types = combined_regimes["VolRegimeType"].unique()
    vol_colors = {"Low": "cyan", "Medium": "orange", "High": "magenta", "Unknown": "lightgray"}
    for regime_type in vol_regime_types:
        mask = (combined_regimes["VolRegimeType"] == regime_type)
        color = vol_colors.get(regime_type, "lightgray")
        ax2.fill_between(combined_regimes.index, 0, 1, where=mask, color=color, alpha=0.3, transform=ax2.get_xaxis_transform())
    ax2.plot(combined_regimes.index, combined_regimes["MeanVolatility"].fillna(0), color="black", linewidth=0.5, label="Mean Volatility (in regime)")
    ax2.set_title(f"{title} - Volatility Regimes")
    ax2.set_ylabel("Volatility Regime")
    ax2.set_yticks([])
    legend_elements_vol = [Patch(facecolor=vol_colors[rt], alpha=0.3, label=rt) for rt in vol_regime_types if rt in vol_colors]
    ax2.legend(handles=legend_elements_vol, loc="upper right")
    ax2.grid(True, axis="x")

    # Plot correlation regimes
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    corr_regime_types = combined_regimes["CorrRegimeType"].unique()
    corr_colors = {"Low": "yellow", "High": "purple", "Unknown": "lightgray"}
    for regime_type in corr_regime_types:
        mask = (combined_regimes["CorrRegimeType"] == regime_type)
        color = corr_colors.get(regime_type, "lightgray")
        ax3.fill_between(combined_regimes.index, 0, 1, where=mask, color=color, alpha=0.3, transform=ax3.get_xaxis_transform())
    ax3.plot(combined_regimes.index, combined_regimes["AvgCorrelation"].fillna(0), color="black", linewidth=0.5, label="Avg Correlation (in regime)")
    ax3.set_title(f"{title} - Correlation Regimes")
    ax3.set_ylabel("Correlation Regime")
    ax3.set_yticks([])
    legend_elements_corr = [Patch(facecolor=corr_colors[rt], alpha=0.3, label=rt) for rt in corr_regime_types if rt in corr_colors]
    ax3.legend(handles=legend_elements_corr, loc="upper right")
    ax3.grid(True, axis="x")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{title.lower().replace(' ', '_')}_combined_regimes.png"))
    plt.close()
    print(f"Saved combined regimes plot to {FIGURES_DIR}")

def main():
    """Main function to run the regime detection analysis."""
    print("Starting regime detection analysis...")
    
    # Load data
    print("Loading processed data...")
    prices_df, returns_df, indices_df, corr_matrix = load_processed_data()
    
    if returns_df.empty:
        print("Returns data is empty. Cannot proceed.")
        return

    # Use a benchmark for market and volatility regimes, e.g., the first asset or a specific index return
    # For simplicity, let's use the returns of the first asset if available, otherwise the first index
    if not returns_df.empty:
        benchmark_returns = returns_df.iloc[:, 0] # Example: first asset's returns
        benchmark_name = returns_df.columns[0].replace("_daily_return", "")
    elif not indices_df.empty:
        benchmark_returns = indices_df.iloc[:, 0] # Example: first index's returns
        benchmark_name = indices_df.columns[0]
    else:
        print("No benchmark returns available for market and volatility regime detection.")
        benchmark_returns = None
        benchmark_name = "Benchmark"

    all_market_regimes = None
    all_vol_regimes = None
    all_corr_regimes = None
    avg_correlations_ts = None

    if benchmark_returns is not None and not benchmark_returns.dropna().empty:
        print(f"Detecting market regimes for {benchmark_name}...")
        all_market_regimes = detect_market_regimes_hmm(benchmark_returns, n_regimes=3)
        plot_market_regimes(benchmark_returns, all_market_regimes, f"{benchmark_name} Market")
        print(f"Market regimes detected and plotted for {benchmark_name}.")
        
        print(f"Detecting volatility regimes for {benchmark_name}...")
        # Calculate rolling volatility for the benchmark
        rolling_vol = benchmark_returns.rolling(window=22).std() * np.sqrt(252) # Annualized rolling volatility (approx 1 month window)
        if not rolling_vol.dropna().empty:
            all_vol_regimes = detect_volatility_regimes(rolling_vol.dropna(), n_regimes=3)
            plot_volatility_regimes(rolling_vol.dropna(), all_vol_regimes, f"{benchmark_name} Volatility")
            print(f"Volatility regimes detected and plotted for {benchmark_name}.")
        else:
            print(f"Could not calculate rolling volatility or it was all NaN for {benchmark_name}.")
    else:
        print("Skipping market and volatility regime detection due to lack of suitable benchmark returns or empty data.")

    if len(returns_df.columns) > 1:
        print("Detecting correlation regimes for the portfolio...")
        all_corr_regimes, avg_correlations_ts = detect_correlation_regimes(returns_df, window=60, n_regimes=2)
        print('testing')
        if avg_correlations_ts is not None and not avg_correlations_ts.empty:
            plot_correlation_regimes(avg_correlations_ts, all_corr_regimes, "Portfolio Correlation")
            print("Correlation regimes detected and plotted for the portfolio.")
        else:
            print("Could not calculate average correlations or data was insufficient.")
    else:
        print("Skipping correlation regime detection due to insufficient assets or returns data.")

    # Combine and plot if all regimes are available
    if all_market_regimes is not None and all_vol_regimes is not None and all_corr_regimes is not None:
        print("Combining all regime information...")
        combined_regimes = combine_regime_information(all_market_regimes, all_vol_regimes, all_corr_regimes)
        if not combined_regimes.empty:
            plot_combined_regimes(combined_regimes, "Portfolio Combined")
            combined_regimes.to_csv(os.path.join(TABLES_DIR, "combined_regime_data.csv"))
            print(f"Combined regime data saved to {TABLES_DIR}")
        else:
            print("Combined regimes DataFrame is empty, skipping combined plot and save.")
    else:
        print("Skipping combined regime analysis as one or more regime types could not be determined.")

    print("Regime detection analysis finished.")

if __name__ == "__main__":
    main()

