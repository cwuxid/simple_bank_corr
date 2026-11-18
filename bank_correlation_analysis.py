#!/usr/bin/env python3
"""Daily bank equity correlation analysis since 2015 using Yahoo Finance.

Install dependencies:
    python -m pip install --upgrade yfinance pandas numpy matplotlib

Run:
    python bank_correlation_analysis.py

Outputs are written to ./bank_correlation_results/.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import yfinance as yf

# BNY replaced BK on May 21, 2026; Yahoo serves its history under BNY.
TICKERS = ["JPM", "BAC", "C", "WFC", "GS", "USB", "COF", "PNC", "TFC", "BNY"]
WINDOW = 90  # Trading days, NOT calendar days.
OUTDIR = Path("bank_correlation_results")


def equal_weight_portfolio_log_returns(log_returns: pd.DataFrame) -> pd.Series:
    # Daily rebalancing averages simple returns before converting back to logs.
    return np.log1p(np.expm1(log_returns).mean(axis=1)).rename("Equal-weight portfolio")


def annual_bank_correlations(log_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means, medians = {}, {}
    for year, annual_returns in log_returns.groupby(log_returns.index.year):
        correlations = annual_returns.corr(method="pearson")
        # Aggregate only the other banks; undefined pair correlations stay undefined.
        peers = {bank: correlations.loc[bank].drop(labels=bank) for bank in correlations.index}
        means[year] = pd.Series({bank: values.mean(skipna=False) for bank, values in peers.items()})
        medians[year] = pd.Series({bank: values.median(skipna=False) for bank, values in peers.items()})
    mean_table = pd.DataFrame(means).rename_axis(index="Bank", columns="Year")
    median_table = pd.DataFrame(medians).rename_axis(index="Bank", columns="Year")
    return mean_table, median_table


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # Fetch extra history so the first date within the reporting period
    # can have a complete 90-trading-day window.
    end_date = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    report_start = pd.Timestamp("2015-01-01")
    annual_end = pd.Timestamp(year=end_date.year, month=1, day=1)
    annual_start = report_start
    download_start = report_start - pd.Timedelta(days=185)
    download_end = end_date + pd.Timedelta(days=1)  # yfinance end is exclusive

    print(f"Downloading prices: {download_start.date()} through {end_date.date()}")
    raw = yf.download(
        tickers=TICKERS,
        start=download_start.strftime("%Y-%m-%d"),
        end=download_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,  # 'Close' is split- and dividend-adjusted
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("No prices downloaded. Check the internet connection and Yahoo Finance availability.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError(f"Cannot locate adjusted Close prices. Columns: {raw.columns}")
        prices = raw["Close"].copy()
    else:
        if len(TICKERS) != 1 or "Close" not in raw.columns:
            raise RuntimeError(f"Unexpected Yahoo Finance columns: {raw.columns}")
        prices = raw[["Close"]].rename(columns={"Close": TICKERS[0]})

    prices = prices.astype(float).replace([np.inf, -np.inf], np.nan)
    # Failed downloads can leave an all-NaN column for the requested ticker.
    missing = [ticker for ticker in TICKERS
               if ticker not in prices.columns or prices[ticker].isna().all()]
    if missing:
        raise RuntimeError(f"No usable prices for tickers: {missing}. Check their symbols and Yahoo Finance availability.")
    prices = prices[TICKERS].sort_index()
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    # Use only days where ALL tickers have prices, ensuring a shared sample.
    prices = prices.dropna(how="any")
    if len(prices) < WINDOW + 10:
        raise RuntimeError(f"Insufficient common price observations: {len(prices)}.")
    if (prices <= 0).any().any():
        raise RuntimeError("Downloaded prices contain zero or negative values.")

    # Daily log returns: ln(P_t/P_{t-1}), expressed in decimal units.
    returns_full = np.log(prices / prices.shift(1)).dropna(how="any")
    prices_period = prices.loc[prices.index >= report_start]
    returns_period = returns_full.loc[returns_full.index >= report_start]
    portfolio_returns = equal_weight_portfolio_log_returns(returns_period)
    annual_returns = returns_full.loc[(returns_full.index >= annual_start) & (returns_full.index < annual_end)]
    annual_mean, annual_median = annual_bank_correlations(annual_returns)

    # Matrix from the reporting sample, excluding the pre-period buffer.
    correlation = returns_period.corr(method="pearson").loc[TICKERS, TICKERS]

    # Rolling Pearson correlation using the past 90 trading-day returns.
    # Keep the full 10x10 matrix, including the redundant lower triangle and diagonal.
    rolling = pd.DataFrame(index=returns_full.index)
    for row in TICKERS:
        for col in TICKERS:
            rolling[f"{row}__{col}"] = returns_full[row].rolling(WINDOW, min_periods=WINDOW).corr(returns_full[col])
    rolling = rolling.loc[rolling.index >= report_start].clip(lower=-1, upper=1)

    prices_period.to_csv(OUTDIR / "adjusted_daily_prices.csv", float_format="%.8f")
    returns_period.to_csv(OUTDIR / "daily_log_returns.csv", float_format="%.10f")
    correlation.to_csv(OUTDIR / "full_period_correlation_matrix.csv", float_format="%.6f")
    rolling.to_csv(OUTDIR / "rolling_90day_correlations.csv", float_format="%.6f")
    annual_mean.to_csv(OUTDIR / "annual_mean_bank_correlations.csv", float_format="%.6f", na_rep="N/A")
    annual_median.to_csv(OUTDIR / "annual_median_bank_correlations.csv", float_format="%.6f", na_rep="N/A")

    fig, axes = plt.subplots(2, 1, figsize=(max(12, 1.25 * len(annual_mean.columns) + 2), 10))
    for ax, table_data, statistic in zip(axes, [annual_mean, annual_median], ["Mean", "Median"]):
        ax.axis("off")
        ax.set_title(f"{statistic} Pearson correlation with the other nine banks", pad=12)
        table = ax.table(
            cellText=table_data.map(lambda value: "N/A" if pd.isna(value) else f"{value:.3f}").to_numpy(),
            rowLabels=table_data.index,
            colLabels=table_data.columns.astype(str),
            cellLoc="center", rowLoc="center", loc="center", bbox=[0.07, 0, 0.93, 0.95],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d5dce5")
            if row == 0:
                cell.set_facecolor("#183a5c")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#eef3f8" if row % 2 else "white")
                if col == -1:
                    cell.set_text_props(weight="bold")
    fig.suptitle(f"Annual bank correlations: {annual_start.year}\N{EN DASH}{annual_end.year - 1}", fontsize=17)
    fig.text(0.5, 0.02, "Complete calendar years; correlations calculated from each year's daily log returns.\n"
             "Each cell summarizes nine pairwise correlations; self-correlation excluded.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.96), h_pad=3)
    fig.savefig(OUTDIR / "annual_bank_correlation_tables.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True, sharey=True)
    axes[0].plot(portfolio_returns.index, portfolio_returns, color="navy", linewidth=0.7)
    axes[0].set_title("Equal-weight portfolio (10% per stock, rebalanced daily)")
    for ticker, color in zip(TICKERS, plt.get_cmap("tab10").colors):
        axes[1].plot(returns_period.index, returns_period[ticker], label=ticker,
                     color=color, linewidth=0.6, alpha=0.7)
    axes[1].set_title("Individual bank equities")
    axes[1].legend(ncol=5, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Daily log return (%)")
    for ax in axes:
        ax.axhline(0, linewidth=0.5, color="gray")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_xlabel("Date")
        ax.grid(alpha=0.15)
    fig.suptitle("Daily log returns: equal-weight bank portfolio and its 10 stocks")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTDIR / "portfolio_vs_stock_daily_log_returns.png", dpi=200)
    plt.close(fig)

    # Heatmap of full-sample correlations.
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(TICKERS)), TICKERS)
    ax.set_yticks(range(len(TICKERS)), TICKERS)
    ax.set_title(f"Bank equity: daily log-return correlations, {report_start.year}\N{EN DASH}{end_date.year}")
    for i in range(len(TICKERS)):
        for j in range(len(TICKERS)):
            val = correlation.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(val) >= 0.65 else "black")
    fig.colorbar(im, ax=ax, label="Pearson correlation", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "full_period_correlation_heatmap.png", dpi=200)
    plt.close(fig)

    # 10-by-10 small-multiple grid, with row stock vs column stock.
    fig, axes = plt.subplots(10, 10, figsize=(26, 23), sharex=True, sharey=True)
    for i, row in enumerate(TICKERS):
        for j, col in enumerate(TICKERS):
            ax = axes[i, j]
            series = rolling[f"{row}__{col}"]
            ax.plot(series.index, series.values, linewidth=0.65)
            ax.axhline(0, linewidth=0.4, color="gray", alpha=0.6)
            ax.set_ylim(-1.05, 1.05)
            if i == 0:
                ax.set_title(col, fontsize=10, pad=7)
            if j == 0:
                ax.set_ylabel(row, fontsize=10, rotation=0, labelpad=23, va="center")
                ax.set_yticks([-1, 0, 1])
            else:
                ax.tick_params(axis="y", labelleft=False)
            if i == 9:
                ax.tick_params(axis="x", labelrotation=40, labelsize=7)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.grid(alpha=0.12)
    fig.suptitle("90-trading-day rolling correlation of bank daily log returns\nRows vs. columns; diagonal = 1", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0.015, 0.015, 1, 0.97), h_pad=0.5, w_pad=0.5)
    fig.savefig(OUTDIR / "rolling_90day_correlation_grid_10x10.png", dpi=160)
    plt.close(fig)

    print(f"\nReturn correlation matrix since {report_start.date()}:")
    print(correlation.round(3).to_string())
    print("\nAnnual mean correlations with the other nine banks (complete calendar years):")
    print(annual_mean.round(3).to_string())
    print("\nAnnual median correlations with the other nine banks (complete calendar years):")
    print(annual_median.round(3).to_string())
    print(f"\nCommon trading sessions since {report_start.date()}: {len(prices_period)}")
    print(f"First/last return dates: {returns_period.index.min().date()} / {returns_period.index.max().date()}")
    print(f"Outputs: {OUTDIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
