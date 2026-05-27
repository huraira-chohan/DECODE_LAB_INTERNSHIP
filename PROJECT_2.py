# ============================================================
# Project 2: Data Classification Using AI
# DecodeLabs Industrial Training Kit | Batch: 2026
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: Load and Understand the Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  PROJECT 2: DATA CLASSIFICATION USING AI")
print("  DecodeLabs | Batch 2026")
print("=" * 60)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("\n[STEP 1] Dataset Overview")
print("-" * 40)
print(f"  Dataset : Iris Flower Dataset")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print(f"  Classes : {list(iris.target_names)}")
print(f"\n  First 5 rows:")
print(df.head().to_string(index=False))

print(f"\n  Statistical Summary:")
print(df.describe().to_string())

print(f"\n  Class Distribution:")
print(df["species"].value_counts().to_string())

# ─────────────────────────────────────────────
# STEP 2: Split Data into Training & Testing Sets
# ─────────────────────────────────────────────
print("\n[STEP 2] Train / Test Split")
print("-" * 40)

X = df[iris.feature_names]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Total samples  : {len(df)}")
print(f"  Training set   : {len(X_train)} samples (80%)")
print(f"  Testing set    : {len(X_test)}  samples (20%)")

# Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# STEP 3: Apply Classification Algorithms
# ─────────────────────────────────────────────
print("\n[STEP 3] Training Classification Models")
print("-" * 40)

models = {
    "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree           ": DecisionTreeClassifier(random_state=42),
    "Logistic Regression     ": LogisticRegression(max_iter=200, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    results[name.strip()] = {
        "model": model,
        "predictions": y_pred,
        "accuracy": acc,
    }
    print(f"  {name}  →  Accuracy: {acc * 100:.2f}%")

# ─────────────────────────────────────────────
# STEP 4: Detailed Report for Best Model
# ─────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["accuracy"])
best = results[best_name]

print(f"\n[STEP 4] Best Model: {best_name}")
print("-" * 40)
print(f"\n  Classification Report:\n")
print(classification_report(y_test, best["predictions"],
                             target_names=iris.target_names))

# ─────────────────────────────────────────────
# STEP 5: Visualisations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Project 2 – Data Classification Using AI\nDecodeLabs | Batch 2026",
             fontsize=13, fontweight="bold", y=1.02)

# --- Plot 1: Class Distribution ---
ax1 = axes[0]
counts = df["species"].value_counts()
colors = ["#4C72B0", "#55A868", "#DD8452"]
bars = ax1.bar(counts.index, counts.values, color=colors, edgecolor="white",
               linewidth=1.2)
ax1.set_title("Class Distribution", fontweight="bold")
ax1.set_xlabel("Species")
ax1.set_ylabel("Count")
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             str(val), ha="center", va="bottom", fontweight="bold")

# --- Plot 2: Feature Pair (Petal Length vs Width) ---
ax2 = axes[1]
palette = {"setosa": "#4C72B0", "versicolor": "#55A868", "virginica": "#DD8452"}
for sp, grp in df.groupby("species"):
    ax2.scatter(grp["petal length (cm)"], grp["petal width (cm)"],
                label=sp, color=palette[sp], alpha=0.7, s=50, edgecolors="white")
ax2.set_title("Feature Scatter: Petal Dimensions", fontweight="bold")
ax2.set_xlabel("Petal Length (cm)")
ax2.set_ylabel("Petal Width (cm)")
ax2.legend(title="Species")

# --- Plot 3: Confusion Matrix for Best Model ---
ax3 = axes[2]
cm = confusion_matrix(y_test, best["predictions"], labels=iris.target_names)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names,
            ax=ax3, linewidths=0.5)
ax3.set_title(f"Confusion Matrix\n({best_name})", fontweight="bold")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("/home/claude/results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[✓] Visualisations saved → results.png")

# ─────────────────────────────────────────────
# STEP 6: Model Accuracy Comparison Chart
# ─────────────────────────────────────────────
names  = [n for n in results]
accs   = [results[n]["accuracy"] * 100 for n in results]

fig2, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(names, accs, color=["#4C72B0", "#55A868", "#DD8452"],
               edgecolor="white", linewidth=1.2)
ax.set_xlim(0, 110)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison\nProject 2 – DecodeLabs",
             fontweight="bold")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig("/home/claude/accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Accuracy comparison chart saved → accuracy_comparison.png")

# ─────────────────────────────────────────────
# STEP 7: Quick Prediction Demo
# ─────────────────────────────────────────────
print("\n[STEP 7] Live Prediction Demo")
print("-" * 40)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # likely setosa
sample_sc = scaler.transform(sample)
best_model = best["model"]
pred = best_model.predict(sample_sc)[0]
print(f"  Input Features (sepal_len, sepal_w, petal_len, petal_w): {sample[0]}")
print(f"  Predicted Species : {pred}")

print("\n" + "=" * 60)
print("  All steps completed successfully!")
print(f"  Best model : {best_name}  ({best['accuracy']*100:.2f}% accuracy)")
print("=" * 60)
