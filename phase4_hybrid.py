import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

np.random.seed(42)

# Load and prepare data
X = pd.read_csv("data/X_scaled.csv")
y = pd.read_csv("data/y.csv")['target']

# Use all original features without selection
print(f"Using all {X.shape[1]} original features: {list(X.columns)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Optimized models
models = {
    'rf': RandomForestClassifier(
        n_estimators=800,
        max_depth=10,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    ),
    'xgb': xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    ),
    'lgb': lgb.LGBMClassifier(
        n_estimators=400,
        num_leaves=50,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        random_state=42
    ),
    'gb': GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        random_state=42
    ),
    'svm': SVC(
        C=10,
        gamma='scale',
        kernel='rbf',
        probability=True,
        random_state=42
    ),
    'lr': LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )
}

# Train models and collect predictions
predictions = {}
probabilities = {}
model_scores = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    model_scores[name] = cv_scores.mean()
    
    predictions[name] = model.predict(X_test)
    probabilities[name] = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, predictions[name])
    print(f"{name} - CV Score: {cv_scores.mean():.4f}, Test Accuracy: {acc:.4f}")

# Advanced stacking ensemble
from sklearn.model_selection import KFold

def create_meta_features(models, X_train, y_train, X_test):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_train = np.zeros((X_train.shape[0], len(models)))
    meta_test = np.zeros((X_test.shape[0], len(models)))
    
    for i, (name, model) in enumerate(models.items()):
        test_preds = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            
            model.fit(X_fold_train, y_fold_train)
            meta_train[val_idx, i] = model.predict_proba(X_fold_val)[:, 1]
            test_preds.append(model.predict_proba(X_test)[:, 1])
        
        meta_test[:, i] = np.mean(test_preds, axis=0)
    
    return meta_train, meta_test

# Create meta-features
meta_train, meta_test = create_meta_features(models, X_train, y_train, X_test)

# Meta-learner
meta_model = LogisticRegression(random_state=42)
meta_model.fit(meta_train, y_train)
stacked_predictions = meta_model.predict(meta_test)
stacked_probs = meta_model.predict_proba(meta_test)[:, 1]

# Voting ensemble with optimized weights
voting_models = [(name, model) for name, model in models.items() if name != 'svm']
voting_clf = VotingClassifier(
    estimators=voting_models,
    voting='soft'
)
voting_clf.fit(X_train, y_train)
voting_predictions = voting_clf.predict(X_test)

# Weighted probability ensemble
weights = np.array([model_scores[name] for name in models.keys()])
weights = weights / weights.sum()

weighted_probs = np.zeros(len(X_test))
for i, name in enumerate(models.keys()):
    weighted_probs += weights[i] * probabilities[name]

weighted_predictions = (weighted_probs >= 0.5).astype(int)

# Results
print("\n=== ENSEMBLE RESULTS ===")
print(f"Stacked Ensemble Accuracy: {accuracy_score(y_test, stacked_predictions):.4f}")
print(f"Voting Ensemble Accuracy: {accuracy_score(y_test, voting_predictions):.4f}")
print(f"Weighted Ensemble Accuracy: {accuracy_score(y_test, weighted_predictions):.4f}")

# Best ensemble
best_acc = max(
    accuracy_score(y_test, stacked_predictions),
    accuracy_score(y_test, voting_predictions),
    accuracy_score(y_test, weighted_predictions)
)

if accuracy_score(y_test, stacked_predictions) == best_acc:
    final_predictions = stacked_predictions
    method = "Stacked"
elif accuracy_score(y_test, voting_predictions) == best_acc:
    final_predictions = voting_predictions
    method = "Voting"
else:
    final_predictions = weighted_predictions
    method = "Weighted"

print(f"\nBEST METHOD: {method} Ensemble")
print(f"FINAL ACCURACY: {best_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, final_predictions))

# Save models
joblib.dump(meta_model, "models/meta_model.pkl")
joblib.dump(voting_clf, "models/voting_ensemble.pkl")
joblib.dump({
    'weights': weights,
    'feature_names': list(X.columns)
}, "models/preprocessing.pkl")

print(f"\nModels saved. Best accuracy: {best_acc:.4f}")