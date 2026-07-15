"""
Daily Logger — Daily Performance Tracker
-------------------------------------------
Run this once per day. Asks about your inputs and 2 outputs,
appends one clean row to your_data.csv.
"""

import csv
import os
from datetime import date

CSV_PATH = "your_data.csv"
COLUMNS = [
    "date", "wake_up_time", "news_read", "focus", "dream_vividness",
    "discipline", "problem_solving_ease", "mood", "energy",
    "productivity_score", "pct_better_than_yesterday",
]


def ask_number(prompt, low, high):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
            if low <= value <= high:
                return value
        except ValueError:
            pass
        print(f"Please enter a number between {low} and {high}.")


def main():
    file_exists = os.path.exists(CSV_PATH)
    today = date.today().isoformat()

    wake_up_time = ask_number("What time did you wake up? (e.g. 7.5 for 7:30am, 24h scale): ", 0, 24)
    news_read = ask_number("How many news articles did you read today? ", 0, 100)
    focus = ask_number("Rate your focus today (1-10): ", 1, 10)
    dream_vividness = ask_number("Rate how much you dreamt / recall dreaming (1-10): ", 1, 10)
    discipline = ask_number("Rate your discipline today (1-10): ", 1, 10)
    problem_solving_ease = ask_number("Rate how easily you thought through problems today (1-10): ", 1, 10)
    mood = ask_number("Rate your mood today (1-10): ", 1, 10)
    energy = ask_number("Rate your energy today (1-10): ", 1, 10)
    productivity_score = ask_number("Rate today's productivity (1-7): ", 1, 7)
    pct_better = ask_number("How much better/worse than yesterday, in percent (-7 to +7): ", -7, 7)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(COLUMNS)
        writer.writerow([
            today, wake_up_time, news_read, focus, dream_vividness,
            discipline, problem_solving_ease, mood, energy,
            productivity_score, pct_better,
        ])

    print(f"\nLogged entry for {today}.")


if __name__ == "__main__":
    main()
