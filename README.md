# Regime Detection Project for European Assets

## Overview

This project implements a regime detection algorithm for a portfolio of European assets. It uses a Hidden Markov Model (HMM) to identify different market regimes (e.g., Bull, Bear, Sideways) based on asset returns. Additionally, it includes methods for detecting volatility regimes using K-means clustering and analyzing correlation regimes.

The project is based on scripts provided by the user, which have been organized, updated, and tested.

## Project Structure

The project is organized as follows:

```
regime_detection_project/
├── data/
│   ├── raw/          # Raw data downloaded (or dummy data)
│   └── processed/    # Processed data ready for analysis
├── results/
│   ├── figures/      # Generated plots and figures
│   └── tables/       # Generated tables and summary statistics
├── src/
│   ├── collect_data.py     # Script to download asset data
│   ├── process_data.py     # Script to process raw data
│   └── regime_detection.py # Main script for regime detection and analysis
├── regime_analysis.ipynb # Jupyter notebook for interactive analysis and visualization
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Installation

1.  **Clone or download the project.**
2.  **Navigate to the project directory:**
    ```bash
    cd regime_detection_project
    ```
3.  **Install the required Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You will also need Jupyter Notebook or JupyterLab to run the `.ipynb` file.* You can install it via pip:
    ```bash
    pip install jupyterlab
    ```

## Usage

There are two main ways to run the analysis:

**Option 1: Using the Jupyter Notebook (Recommended)**

1.  **Start JupyterLab:**
    ```bash
    jupyter lab
    ```
2.  **Open `regime_analysis.ipynb`:** Navigate to the notebook file in the JupyterLab interface and open it.
3.  **Run the cells:** Execute the cells sequentially. The notebook explains the intuition behind each step, runs the data processing and regime detection scripts, and displays the resulting plots directly.

**Option 2: Running Python Scripts Directly**

1.  **Data Collection (Known Limitation):**
    The `src/collect_data.py` script is intended to download historical data for European ETFs and indices using `yfinance`. However, during testing, `yfinance` failed to retrieve data for the specified European symbols (error: "No timezone found, symbol may be delisted").
    *   **Current State:** To allow the project pipeline to run, dummy data files (`*.csv`) have been placed in the `data/raw/` directory. These files contain placeholder data for a short period.
    *   **Future Work:** To use live data, you may need to find alternative, valid ticker symbols for European assets compatible with `yfinance` or use a different data provider/API and modify `src/collect_data.py` accordingly. If you modify the script, run it:
        ```bash
        python src/collect_data.py
        ```

2.  **Data Processing:**
    Run the data processing script. This will load the raw (dummy or real) data, clean it, calculate returns, and save the processed files (`asset_prices.csv`, `asset_returns.csv`, etc.) into the `data/processed/` directory.
    ```bash
    python src/process_data.py
    ```

3.  **Regime Detection:**
    Run the main regime detection script. This script loads the processed data, trains the HMM and K-means models, identifies market, volatility, and correlation regimes, generates plots (saved in `results/figures/`), and potentially saves model files (in `models/`) and summary tables (in `results/tables/`).
    ```bash
    python src/regime_detection.py
    ```

## Outputs

*   **Processed Data:** Located in `data/processed/`. Includes asset prices, returns, correlations, etc.
*   **Models:** Trained models (like HMM) might be saved in the `models/` directory.
*   **Results:** Figures visualizing the detected regimes are saved in `results/figures/`. Summary statistics or tables might be saved in `results/tables/`.
*   **Notebook:** `regime_analysis.ipynb` provides an interactive walkthrough of the process and displays results inline.

## Known Issues & Limitations

*   **Data Collection:** As mentioned, `yfinance` currently fails to download data for the specified European symbols. The project uses dummy data in `data/raw/` for demonstration and testing purposes. The results generated will be based on this limited dummy data.

