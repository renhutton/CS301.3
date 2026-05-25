import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────

CSV_PATH    = "gesture_data.csv"
MODEL_PATH  = "gesture_model.pkl"
NUM_FEATURES = 126          # 63 per hand × 2 hands
TEST_SIZE    = 0.2
RANDOM_SEED  = 42

# ── Load & validate data ──────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH, header=None)

print(f"Total samples:   {len(df)}")
print(f"Total columns:   {len(df.columns)}  (expected {NUM_FEATURES + 1})")
print(f"\nSamples per gesture:")
print(df[0].value_counts())

# Sanity check
assert len(df.columns) == NUM_FEATURES + 1, (
    f"Expected {NUM_FEATURES + 1} columns (1 label + {NUM_FEATURES} features), "
    f"got {len(df.columns)}. Check your collection script."
)

# Check for missing values
if df.isnull().any().any():
    print(f"\nWarning: found {df.isnull().sum().sum()} missing values — dropping those rows")
    df = df.dropna()

X = df.iloc[:, 1:].values.astype(np.float32)
y = df.iloc[:, 0].values

# ── Encode labels ─────────────────────────────────────────────────────────────

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nGesture classes: {list(le.classes_)}")

# ── Train / test split ────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_encoded      # keeps class balance in both splits
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

# ── Scale features ────────────────────────────────────────────────────────────
# Normalised landmarks are already roughly in the same range,
# but scaling still helps the MLP converge faster and more reliably

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)     # use train stats — never fit on test data

# ── Train model ───────────────────────────────────────────────────────────────

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # slightly deeper for 126 input features
    activation="relu",
    solver="adam",
    alpha=0.001,                        # L2 regularisation — helps avoid overfitting
    learning_rate="adaptive",
    max_iter=1000,
    random_state=RANDOM_SEED,
    early_stopping=True,                # stops if val loss stops improving
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=True
)

print("\nTraining...")
model.fit(X_train, y_train)
print(f"Stopped after {model.n_iter_} iterations")

# ── Evaluate ──────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)

print("\n── Classification Report ──────────────────────────────")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Overall accuracy
accuracy = (y_pred == y_test).mean()
print(f"Test accuracy: {accuracy * 100:.1f}%")

# ── Plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=axes[0]
)
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted Label")

# Loss curve

axes[1].plot(model.loss_curve_, label="Training loss")
if model.best_loss_ is not None:
    axes[1].plot(model.best_loss_ * np.ones(len(model.loss_curve_)),
                 "r--", label=f"Best val loss: {model.best_loss_:.4f}")
axes[1].set_title("Training Loss Curve")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
print("Saved training_results.png")

# ── Save model ────────────────────────────────────────────────────────────────

with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "model":   model,
        "scaler":  scaler,
        "encoder": le,
        "num_features": NUM_FEATURES
    }, f)

print(f"\nModel saved to {MODEL_PATH}")
print(f"Classes: {list(le.classes_)}")
