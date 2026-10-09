import pandas as pd

# Load the Iris dataset
df = pd.read_csv("data/IRIS.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display column names
print("\nColumn names:")
print(df.columns)

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Display missing values
print("\nMissing values:")
print(df.isnull().sum())

print("\nMissing values:")
print(df.isnull().sum())



import matplotlib.pyplot as plt
import seaborn as sns

print("\nSpecies count:")
print(df["species"].value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="species")
plt.title("Iris Species Distribution")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

sns.pairplot(df, hue="species")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(
    df.drop("species", axis=1).corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation")
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create models
models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}


# Train and evaluate models
for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ==============================
# MODEL COMPARISON
# ==============================

import joblib

# Create models
models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

results = {}

best_model = None
best_model_name = ""
best_accuracy = 0

# Train and evaluate each model
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Check best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name


# ==============================
# MODEL COMPARISON RESULTS
# ==============================

print("\n\nMODEL COMPARISON")
print("==============================")

for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")


# ==============================
# SAVE BEST MODEL
# ==============================

joblib.dump(best_model, "best_model.joblib")

print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)

print("\nBest model saved as: best_model.joblib")