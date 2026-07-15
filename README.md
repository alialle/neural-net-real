# Neural Net Playground

A personal project for tracking daily life challenges, logging real data about them, and training small neural networks to find patterns — built while learning neural network concepts hands-on (epochs, loss, overfitting, hidden layers) using Google's TensorFlow Playground as a starting point.

Each "challenge" is a self-contained mini-project: its own inputs, its own CSV of logged data, its own neural network, and its own charts. Unrelated challenges get separate networks rather than being mixed into one — this keeps each model focused and easy to reason about.

## Project structure

```
nnw/
├── README.md
├── nn_template.py              # generic reusable regression template
├── nn_template_classifier.py   # generic reusable classification template
└── challenges/
    ├── sleep_music_coffee/
    │   ├── log_entry.py         # daily interactive logger
    │   ├── view_chart.py        # trend charts
    │   ├── nn_template.py       # trained on this challenge's data
    │   └── your_data.csv        # local only, not pushed to GitHub
    └── daily_performance/
        ├── log_entry.py
        ├── view_chart.py         # trend charts + exponential fit
        ├── nn_template_multi.py  # multi-output neural net
        └── your_data.csv         # local only, not pushed to GitHub
```

## Challenges tracked

### 1. Sleep / Music / Coffee → Productivity
- **Inputs:** listened_to_music, drank_coffee, hours_slept
- **Output:** productivity_score (1-10)
- Single-output regression (`MLPRegressor`)

### 2. Daily Performance
- **Inputs:** wake_up_time, news_read, focus, dream_vividness, discipline, problem_solving_ease, mood, energy
- **Outputs (2, predicted at once):** productivity_score (1-7), pct_better_than_yesterday (-7 to +7)
- Multi-output regression — one network, two outputs sharing the same hidden layers
- Chart view includes a fitted exponential trend line per output (needs 3+ logged days)

## Core features (every challenge)

- Interactive daily logger (`log_entry.py`) — asks plain-English questions, appends a clean row to CSV, no manual typing of commas
- 80/20 train/test split
- 2 hidden layers, ReLU activation
- Epoch-by-epoch checkpointing — automatically keeps the model version with the **lowest test loss seen**, not just whatever the last epoch produced
- Clear side-by-side print of training loss vs test loss, to catch overfitting
- Chart viewer (`view_chart.py`) — visualizes trends and habit comparisons
- Ready-to-edit prediction section at the bottom of each training script

## Setup

```bash
pip3 install pandas scikit-learn matplotlib scipy
```

## Daily workflow

```bash
cd ~/Downloads/nnw/challenges/<challenge_name>
python3 log_entry.py        # log today's entry
python3 view_chart.py       # see trends (any time)
python3 nn_template.py      # train the model (once you have 10-15+ entries)
```

## Adding a new challenge

```bash
cd ~/Downloads/nnw
mkdir -p challenges/new_challenge_name
cp nn_template.py challenges/new_challenge_name/
```
Then edit `CSV_PATH`, `INPUT_COLUMNS`, and `OUTPUT_COLUMN` (or `OUTPUT_COLUMNS` for multi-output) inside the copied script to match the new challenge's data.

## Data privacy

Real logged data (`your_data.csv` in every challenge folder) is excluded from git via `.gitignore` and never pushed to GitHub — only the scripts and structure are public.

## Roadmap / ideas for later

- [ ] Add cross-validation instead of a single train/test split
- [ ] Add a config file (YAML/JSON) instead of editing scripts directly
- [ ] Add more challenges as new personal patterns come up
- [ ] Compare predicted vs actual on the chart, not just raw trend
