"""
Reusable Neural Network Training Template (Regression)
--------------------------------------------------------
Uses scikit-learn's MLPRegressor to predict a NUMBER from your own
CSV data (e.g. exported from SQL). Rename the placeholders below to
match your actual column names and you're ready to go.
"""

import copy
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. CONFIG — edit these to match your CSV
# ============================================================
CSV_PATH = "your_data.csv"          # path to your exported CSV file
INPUT_COLUMNS = ["listened_to_music", "drank_coffee", "hours_slept"]  # your input columns
OUTPUT_COLUMN = "productivity_score"      # the number you're trying to predict

# ------------------------------------------------------------
# Hyperparameters — the "dials" that control how the network
# is built and trained. Sensible starting defaults are set here.
# ------------------------------------------------------------
HIDDEN_LAYER_SIZES = (8, 4)   # 2 hidden layers: 8 neurons, then 4 neurons
ACTIVATION = "relu"           # activation function each neuron uses
LEARNING_RATE_INIT = 0.03     # how big a step the network takes per update
TEST_SIZE = 0.2               # 80/20 train/test split
MAX_ITER = 2000               # max number of "epochs" (passes over training data)

# ============================================================
# 2. LOAD DATA
# ============================================================
# pandas reads your CSV straight into a table (DataFrame).
# Each row = one training example, each column = one variable.
df = pd.read_csv(CSV_PATH, )
X = df[INPUT_COLUMNS].to_numpy(dtype=float)  # inputs the network learns from
y = df[OUTPUT_COLUMN].to_numpy(dtype=float)  # the number it's trying to predict

# ============================================================
# 3. TRAIN / TEST SPLIT (80/20)
# ============================================================
# We hold back 20% of the data the model never trains on.
# This lets us check if it actually learned the pattern, or just
# memorized the training rows (a problem called "overfitting").
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

# ============================================================
# 4. BUILD & TRAIN THE NETWORK (with checkpointing)
# ============================================================
# hidden_layer_sizes -> how many hidden layers/neurons the network has.
# activation -> the function each neuron uses to fire.
# warm_start=True lets us train one epoch at a time in a loop below,
# instead of all at once, so we can check test loss after every epoch.
model = MLPRegressor(
    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
    activation=ACTIVATION,
    learning_rate_init=LEARNING_RATE_INIT,
    max_iter=1,
    warm_start=True,
    random_state=42,
)

# Checkpointing: as training progresses, test loss can start going back
# UP even while training loss keeps going down. That's overfitting — the
# model is memorizing the training rows instead of learning the general
# pattern. So instead of keeping whatever the LAST epoch produced, we
# keep a saved copy of whichever epoch had the LOWEST test loss so far.
best_test_loss = float("inf")
best_epoch = 0
best_model = None

for epoch in range(1, MAX_ITER + 1):
    with warnings.catch_warnings():
        # We're deliberately training 1 epoch at a time, so sklearn's
        # "hasn't converged yet" warning is expected here — silence it.
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)  # one more epoch (warm_start keeps old weights)
    test_loss_this_epoch = mean_squared_error(y_test, model.predict(X_test))

    if test_loss_this_epoch < best_test_loss:
        best_test_loss = test_loss_this_epoch
        best_epoch = epoch
        best_model = copy.deepcopy(model)  # checkpoint: save this version

print(f"Best test loss (MSE): {best_test_loss:.4f}  (found at epoch {best_epoch} of {MAX_ITER})")

# From here on, use the CHECKPOINTED best model, not the final-epoch model.
model = best_model

# ============================================================
# 5. EVALUATE — training loss vs test loss, side by side
# ============================================================
# "Loss" here = mean squared error (avg squared difference between
# predicted and actual values). Lower is better.
# A big gap between training loss and test loss is a sign of
# overfitting: the model memorized training data instead of
# learning the general pattern.
train_loss = mean_squared_error(y_train, model.predict(X_train))
test_loss = mean_squared_error(y_test, model.predict(X_test))
print(f"Training loss (MSE): {train_loss:.4f}   |   Test loss (MSE): {test_loss:.4f}   |   Gap: {abs(train_loss - test_loss):.4f}")

# ============================================================
# 6. MAKE A PREDICTION — plug in new values here
# ============================================================
new_input = [[0, 0, 0]]  # replace with real values, matching INPUT_COLUMNS order
prediction = model.predict(new_input)
print(f"Prediction for {new_input}: {prediction[0]:.4f}")
