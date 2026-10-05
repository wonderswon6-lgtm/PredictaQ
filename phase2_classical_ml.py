# phase2_classical_ml.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # for saving models

# -------------------------
# Step 0: Load preprocessed data
# -------------------------
X_scaled = pd.read_csv("data/X_scaled.csv")  # scaled features from Phase 1
y = pd.read_csv("data/y.csv")['target']      # target column

# -------------------------
# Step 1: Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------
# Step 2: Logistic Regression
# -------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Save Logistic Regression model
joblib.dump(lr_model, "models/logistic_regression_model.pkl")

# -------------------------
# Step 3: Random Forest
# -------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Save Random Forest model
joblib.dump(rf_model, "models/random_forest_model.pkl")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_scaled.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
feature_importance.to_csv("models/random_forest_feature_importance.csv", index=False)
print("Random Forest Feature Importance saved.")

# -------------------------
# Step 4: Support Vector Machine
# -------------------------
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("===== SVM =====")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Save SVM model
joblib.dump(svm_model, "models/svm_model.pkl")

print("All models saved in 'models/' folder.")
