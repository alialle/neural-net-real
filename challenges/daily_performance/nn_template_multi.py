"""
Reusable Neural Network Training Template (Multi-Output Regression)
-----------------------------------------------------------------------
Uses scikit-learn's MLPRegressor to predict TWO numbers at once from
your daily inputs:
  1. productivity_score      (1-7)
  2. pct_better_than_yesterday (-7 to +7)

A neural network can output multiple numbers from the same hidden
layers — it just needs multiple output columns, which is what
"multi-output regression" means here.
"""

import copy
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. CONFIG
# ============================================================
CSV_PATH = "your_data.csv"
INPUT_COLUMNS = [
    "wake_up_time", "news_read", "focus", "dream_vividness",
    "discipline", "problem_solving_ease", "mood", "energy",
]
OUTPUT_COLUMNS = ["productivity_score", "pct_better_than_yesterday"]  # 2 outputs at once

# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
HIDDEN_LAYER_SIZES = (8, 4)   # 2 hidden layers: 8 neurons, then 4 neurons
ACTIVATION = "relu"
LEARNING_RATE_INIT = 0.03
TEST_SIZE = 0.2               # 80/20 train/test split
MAX_ITER = 2000

# ============================================================
# 2. LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH, dtype_backend="numpy_nullable")
X = df[INPUT_COLUMNS].to_numpy(dtype=float)
y = df[OUTPUT_COLUMNS].to_numpy(dtype=float)  # 2D: one column per output

# ============================================================
# 3. TRAIN / TEST SPLIT (80/20)
# ============================================================
# We hold back 20% of the data the model never trains on, so we can
# check if it learned the real pattern or just memorized rows
# (overfitting).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

# ============================================================
# 4. BUILD & TRAIN THE NETWORK (with checkpointing)
# ============================================================
# warm_start=True lets us train one epoch at a time in the loop below,
# so we can check test loss after every epoch and keep the best one.
model = MLPRegressor(
    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
    activation=ACTIVATION,
    learning_rate_init=LEARNING_RATE_INIT,
    max_iter=1,
    warm_start=True,
    random_state=42,
)

best_test_loss = float("inf")
best_epoch = 0
best_model = None

for epoch in range(1, MAX_ITER + 1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    test_loss_this_epoch = mean_squared_error(y_test, model.predict(X_test))

    if test_loss_this_epoch < best_test_loss:
        best_test_loss = test_loss_this_epoch
        best_epoch = epoch
        best_model = copy.deepcopy(model)  # checkpoint: save this version

print(f"Best test loss (MSE, averaged across both outputs): {best_test_loss:.4f}  (found at epoch {best_epoch} of {MAX_ITER})")

model = best_model

# ============================================================
# 5. EVALUATE — training loss vs test loss, per output
# ============================================================
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_loss_productivity = mean_squared_error(y_train[:, 0], train_pred[:, 0])
test_loss_productivity = mean_squared_error(y_test[:, 0], test_pred[:, 0])
train_loss_pct = mean_squared_error(y_train[:, 1], train_pred[:, 1])
test_loss_pct = mean_squared_error(y_test[:, 1], test_pred[:, 1])

print(f"\nProductivity score  -> Train loss: {train_loss_productivity:.4f}  |  Test loss: {test_loss_productivity:.4f}")
print(f"Pct better/worse     -> Train loss: {train_loss_pct:.4f}  |  Test loss: {test_loss_pct:.4f}")

# ============================================================
# 6. MAKE A PREDICTION — plug in new values here
# ============================================================
# Order must match INPUT_COLUMNS:
# [wake_up_time, news_read, focus, dream_vividness, discipline,
#  problem_solving_ease, mood, energy]
new_input = [[7.5, 3, 6, 5, 6, 6, 7, 6]]
prediction = model.predict(new_input)
print(f"\nPrediction for {new_input}:")
print(f"  Productivity score:        {prediction[0][0]:.2f}")
print(f"  Pct better than yesterday: {prediction[0][1]:.2f}%")
