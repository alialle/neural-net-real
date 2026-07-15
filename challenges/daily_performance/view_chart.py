"""
Chart Viewer — Daily Performance Tracker
--------------------------------------------
Plots your two outputs over time, PLUS a fitted exponential trend
line for each, so you can see whether you're accelerating,
staying flat, or declining over time.

Note: the neural network itself doesn't calculate "exponential
growth" — that's not what hidden layers do. This script instead
fits a standard exponential curve (y = a * e^(b*x)) to your
logged history and draws it alongside your real data points.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CSV_PATH = "your_data.csv"


def exponential_func(x, a, b):
    return a * np.exp(b * x)


def plot_with_trend(ax, dates, values, title, ylabel):
    days_since_start = (dates - dates.min()).dt.days.to_numpy(dtype=float)
    ax.scatter(dates, values, label="Actual", color="steelblue")

    # Exponential fit needs positive values, so shift data up if needed
    shift = 0
    if values.min() <= 0:
        shift = abs(values.min()) + 1
    shifted_values = values + shift

    # Fitting a curve with 2 parameters (a, b) needs at least 3 data
    # points, or the math has more unknowns than equations to solve.
    if len(days_since_start) < 3:
        ax.text(0.5, 0.5, f"Need at least 3 logged days\nfor a trend fit (you have {len(days_since_start)})",
                transform=ax.transAxes, ha="center", va="center", color="gray")
    else:
        try:
            params, _ = curve_fit(
                exponential_func, days_since_start, shifted_values,
                p0=(1, 0.01), maxfev=5000
            )
            fitted = exponential_func(days_since_start, *params) - shift
            ax.plot(dates, fitted, color="orange", linestyle="--", label="Exponential trend")
        except RuntimeError:
            ax.text(0.5, 0.5, "Not enough of a pattern yet\nfor a trend fit",
                    transform=ax.transAxes, ha="center", va="center", color="gray")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()


df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

plot_with_trend(
    axes[0], df["date"], df["productivity_score"],
    "Productivity Score Over Time (1-7)", "Productivity"
)
plot_with_trend(
    axes[1], df["date"], df["pct_better_than_yesterday"],
    "% Better/Worse vs Yesterday", "Percent (-7 to +7)"
)

plt.tight_layout()
plt.savefig("performance_charts.png", dpi=150)
print("Saved chart as performance_charts.png — open it to view.")
plt.show()
