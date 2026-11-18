# simple_bank_corr
A simple script that loads stock data for banks and tracks the correlation over periods as a time series to measure systematic risk. Done for CPG FA2026 policy proposal "Lend Short &amp; Live Long: Testing Emergency Liquidity Readiness for US Banks".

Run `.venv/bin/python bank_correlation_analysis.py` to write results to `bank_correlation_results/`.

Prices, daily returns, portfolio plots, and 90-trading-day rolling correlations cover January 1, 2015 through the latest available data. The full-period correlation matrix and heatmap are saved as `full_period_correlation_matrix.csv` and `full_period_correlation_heatmap.png`.

The annual mean and median correlation CSV tables and `annual_bank_correlation_tables.png` use complete calendar years starting in 2015 (2015–2025 when run in 2026). Rows are banks and columns are years. For each year, the script calculates Pearson correlations from that year's daily log returns, then takes each bank's mean and median correlation with the other nine banks, excluding itself. These are calculated separately from the 90-day rolling correlations. An undefined pairwise correlation produces `N/A` for the affected bank's annual summary.
