import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Input, BatchNormalization, ReLU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2

import optuna
import random
import tensorflow as tf

# ------------------------ Reproducibility ------------------------ #
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------ Generator Class ------------------------ #
class GeneExpressionSequence(Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X.reshape(-1, X.shape[1], 1)
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.X[batch_idx], self.y[batch_idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ------------------------ Model Builders ------------------------ #
def build_model_1layer(trial, input_shape):
    model = Sequential([Input(shape=input_shape)])
    reg = l2(trial.suggest_float("l2_strength", 1e-3, 1e-1, log=True))

    model.add(Conv1D(trial.suggest_categorical("num_filters", [32, 64, 128]),
                     trial.suggest_categorical("kernel_size", [4, 8, 12]),
                     strides=trial.suggest_categorical("conv_stride", [1, 2]),
                     activation='relu', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(trial.suggest_categorical("pool_size", [1, 2, 3]),
                           strides=trial.suggest_categorical("pool_stride", [1, 2, 3])))
    model.add(Flatten())
    model.add(Dense(trial.suggest_categorical("dense_units", [64, 128, 256]), activation='relu', kernel_regularizer=reg))
    model.add(Dropout(trial.suggest_categorical("dropout_rate", [0.07, 0.1])))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_model_3layer(trial, input_shape):
    model = Sequential([Input(shape=input_shape)])
    reg = l2(trial.suggest_float("l2_strength", 1e-3, 1e-1, log=True))
    ks = trial.suggest_categorical("kernel_size", [4, 8, 12])
    stride = trial.suggest_categorical("conv_stride", [1, 2])

    for filters in [8, 16, 32]:
        model.add(Conv1D(filters, ks, strides=stride, padding="same", kernel_regularizer=reg))
        model.add(BatchNormalization())
        model.add(ReLU())

    model.add(MaxPooling1D(trial.suggest_categorical("pool_size", [1, 2, 3]),
                           strides=trial.suggest_categorical("pool_stride", [1, 2, 3])))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=reg))
    model.add(Dropout(trial.suggest_float("dropout_rate", 0.1, 0.3)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------------ Main Execution ------------------------ #
def main(split, batch_size, model_type):
    base = Path("data/data_splits") / split
    X_train = pd.read_csv(base / "X_train.csv", index_col=0).values
    y_train = pd.read_csv(base / "y_train.csv", index_col=0).squeeze().values
    X_test = pd.read_csv(base / "X_test.csv", index_col=0).values
    y_test = pd.read_csv(base / "y_test.csv", index_col=0).squeeze().values

    X_val, X_final, y_val, y_final = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=SEED)
    input_shape = (X_train.shape[1], 1)

    def objective(trial):
        build_fn = build_model_3layer if model_type == "3layer" else build_model_1layer
        model = build_fn(trial, input_shape)
        train_gen = GeneExpressionSequence(X_train, y_train, batch_size=batch_size)
        val_gen = GeneExpressionSequence(X_val, y_val, batch_size=batch_size, shuffle=False)

        model.fit(train_gen, validation_data=val_gen, epochs=30,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                  verbose=0)

        y_pred = (model.predict(val_gen) > 0.5).astype(int).flatten()
        return accuracy_score(y_val, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    model = (build_model_3layer if model_type == "3layer" else build_model_1layer)(study.best_trial, input_shape)
    train_gen = GeneExpressionSequence(X_train, y_train, batch_size=batch_size)
    val_gen = GeneExpressionSequence(X_val, y_val, batch_size=batch_size)
    model.fit(train_gen, validation_data=val_gen, epochs=50,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=1)

    final_gen = GeneExpressionSequence(X_final, y_final, batch_size=16, shuffle=False)
    y_pred_final = (model.predict(final_gen) > 0.5).astype(int).flatten()

    # -------------------- Output & Plots -------------------- #
    out_dir = Path(f"results/{split}/batch{batch_size}/{model_type}")
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_final, y_pred_final)
    report = classification_report(y_final, y_pred_final, output_dict=False, zero_division=0)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"Best Trial Params:\n{study.best_trial.params}\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n{report}\n")

    df_report = pd.DataFrame(classification_report(y_final, y_pred_final, output_dict=True)).T
    df_report.to_csv(out_dir / "report.csv")
    pd.DataFrame(confusion_matrix(y_final, y_pred_final)).to_csv(out_dir / "confusion_matrix.csv")

    sns.heatmap(confusion_matrix(y_final, y_pred_final), annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

# ------------------------ CLI Entry ------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--model_type", type=str, choices=["1layer", "3layer"], required=True)
    args = parser.parse_args()

    main(args.split, args.batch_size, args.model_type)
