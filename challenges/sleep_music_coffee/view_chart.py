"""
Chart Viewer — see your logged data visually
----------------------------------------------
Run this any time to see charts of your logged entries:
  1. Productivity over time
  2. Sleep hours vs productivity (does more sleep = more productive?)
  3. Coffee/music days vs no-coffee/no-music days (average productivity)
"""

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "your_data.csv"

df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Productivity over time
axes[0].plot(df["date"], df["productivity_score"], marker="o")
axes[0].set_title("Productivity Over Time")
axes[0].set_ylabel("Productivity (1-10)")
axes[0].tick_params(axis="x", rotation=45)

# 2. Sleep vs productivity (scatter)
axes[1].scatter(df["hours_slept"], df["productivity_score"])
axes[1].set_title("Sleep Hours vs Productivity")
axes[1].set_xlabel("Hours Slept")
axes[1].set_ylabel("Productivity (1-10)")

# 3. Average productivity: coffee days vs non-coffee days, music vs no music
avg_coffee = df.groupby("drank_coffee")["productivity_score"].mean()
avg_music = df.groupby("listened_to_music")["productivity_score"].mean()
labels = ["No Coffee", "Coffee", "No Music", "Music"]
values = [
    avg_coffee.get(0, 0), avg_coffee.get(1, 0),
    avg_music.get(0, 0), avg_music.get(1, 0),
]
axes[2].bar(labels, values, color=["gray", "brown", "gray", "green"])
axes[2].set_title("Avg Productivity by Habit")
axes[2].set_ylabel("Avg Productivity (1-10)")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("productivity_charts.png", dpi=150)
print("Saved chart as productivity_charts.png — open it to view.")
plt.show()
