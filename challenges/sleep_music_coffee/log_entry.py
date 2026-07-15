"""
Daily Logger — Sleep / Music / Coffee -> Productivity
------------------------------------------------------
Run this once per day. It asks you a few quick questions and
appends one clean row to your_data.csv — no manual typing of
commas or risk of breaking the file format.
"""

import csv
import os
from datetime import date

CSV_PATH = "your_data.csv"
COLUMNS = ["date", "listened_to_music", "drank_coffee", "hours_slept", "productivity_score"]


def ask_yes_no(prompt):
    while True:
        answer = input(prompt + " (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return 1
        if answer in ("n", "no"):
            return 0
        print("Please type y or n.")


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
    music = ask_yes_no("Did you listen to music today?")
    coffee = ask_yes_no("Did you drink coffee today?")
    sleep = ask_number("How many hours did you sleep? ", 0, 24)
    productivity = ask_number("Rate your productivity today (1-10): ", 1, 10)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(COLUMNS)  # write header only if file is new
        writer.writerow([today, music, coffee, sleep, productivity])

    print(f"\nLogged: {today} | music={music} coffee={coffee} sleep={sleep}h productivity={productivity}")


if __name__ == "__main__":
    main()
