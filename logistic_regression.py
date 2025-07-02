import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_score, recall_score
)

# --- Load Dataset ---
df = pd.read_csv("data.csv")
print(" Dataset loaded.\n")

# --- Set Target Column & Clean Up ---
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Encode labels
df = df.drop(columns=['Unnamed: 32'], errors='ignore')  # Drop empty col

# --- Split Features & Target ---
target_col = 'diagnosis'
X = df.drop(columns=[target_col])
y = df[target_col]

# --- Encode (if any), then Impute Missing Values ---
cat_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Fill missing values with mean
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# --- Standardize Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- Train Logistic Regression Model ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f" ROC-AUC Score: {roc_auc:.4f}")

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# --- Sigmoid Function Plot ---
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(6, 4))
plt.plot(z, sigmoid, label="Sigmoid(z)")
plt.axhline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.axvline(0, color='gray', linestyle='--')
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Sigmoid Output")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sigmoid_plot.png")
plt.show()

# --- Custom Threshold Example ---
threshold = 0.6
custom_pred = (y_prob >= threshold).astype(int)
print(f"\n Custom Threshold = {threshold}")
print(f"Precision: {precision_score(y_test, custom_pred):.2f}")
print(f"Recall: {recall_score(y_test, custom_pred):.2f}")
