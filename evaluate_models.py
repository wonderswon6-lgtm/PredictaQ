"""
evaluate_models.py - Evaluate performance of all trained models in models/
"""

import os
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def main():
    print("=" * 70)
    print("           PREDICTA-Q: MODEL PERFORMANCE EVALUATION")
    print("=" * 70)

    # 1. Load Data
    if not os.path.exists("data/X_scaled.csv") or not os.path.exists("data/y.csv"):
        print("[-] Data files not found in data/. Run phase1_data_prep.py first.")
        return

    X = pd.read_csv("data/X_scaled.csv")
    y = pd.read_csv("data/y.csv")['target']

    # Standard split consistent with training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    # 2. Evaluate Classical Pickle Models
    model_files = {
        "Logistic Regression": "models/logistic_regression_model.pkl",
        "Random Forest": "models/random_forest_model.pkl",
        "Support Vector Machine (SVM)": "models/svm_model.pkl",
        "Voting Ensemble": "models/voting_ensemble.pkl",
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                y_pred = model.predict(X_test)
                
                # Check for probability support
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                else:
                    auc = np.nan

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                results.append({
                    "Model": name,
                    "Accuracy": f"{acc * 100:.2f}%",
                    "Precision": f"{prec * 100:.2f}%",
                    "Recall": f"{rec * 100:.2f}%",
                    "F1-Score": f"{f1 * 100:.2f}%",
                    "ROC-AUC": f"{auc:.4f}" if not np.isnan(auc) else "N/A"
                })
            except Exception as e:
                print(f"[!] Warning: Could not evaluate {name}: {e}")

    # 3. Evaluate Quantum Hybrid Model (PyTorch) if present
    vqc_path = "models/hybrid_vqc_model_best.pth"
    if os.path.exists(vqc_path):
        try:
            # Import architecture from phase3
            import pennylane as qml
            import torch.nn as nn

            n_qubits = X_train.shape[1]
            num_layers = 3
            dev = qml.device("default.qubit", wires=n_qubits)

            def circuit(inputs, weights):
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)
                    qml.RZ(inputs[i], wires=i)
                for l in range(num_layers):
                    for i in range(n_qubits):
                        qml.RY(weights[l*2*n_qubits + i], wires=i)
                        qml.RZ(weights[l*2*n_qubits + i + n_qubits], wires=i)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            weight_shapes = {"weights": 2 * n_qubits * num_layers}
            qnode = qml.QNode(circuit, dev, interface="torch")

            class QuantumLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
                def forward(self, x):
                    return self.qlayer(x)

            class HybridModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.pre_fc = nn.Linear(n_qubits, 64)
                    self.fc1 = nn.Linear(64, n_qubits)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.2)
                    self.quantum = QuantumLayer()
                    self.fc2 = nn.Linear(n_qubits, 32)
                    self.fc3 = nn.Linear(32, 1)
                def forward(self, x):
                    x = self.dropout(self.relu(self.pre_fc(x)))
                    x = self.relu(self.fc1(x))
                    x = self.quantum(x)
                    x = self.dropout(self.relu(self.fc2(x)))
                    return self.fc3(x)

            q_model = HybridModel()
            state_dict = torch.load(vqc_path, map_location=torch.device('cpu'))
            q_model.load_state_dict(state_dict)
            q_model.eval()

            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            with torch.no_grad():
                logits = q_model(X_test_tensor).squeeze()
                probs = torch.sigmoid(logits).numpy()
                preds = (probs >= 0.5).astype(int)

            q_acc = accuracy_score(y_test, preds)
            q_prec = precision_score(y_test, preds, zero_division=0)
            q_rec = recall_score(y_test, preds, zero_division=0)
            q_f1 = f1_score(y_test, preds, zero_division=0)
            q_auc = roc_auc_score(y_test, probs)

            results.append({
                "Model": "Hybrid VQC (Quantum)",
                "Accuracy": f"{q_acc * 100:.2f}%",
                "Precision": f"{q_prec * 100:.2f}%",
                "Recall": f"{q_rec * 100:.2f}%",
                "F1-Score": f"{q_f1 * 100:.2f}%",
                "ROC-AUC": f"{q_auc:.4f}"
            })
        except Exception as e:
            print(f"[!] Info: Quantum VQC evaluation skipped: {e}")

    # 4. Display Results Table
    if results:
        df_results = pd.DataFrame(results)
        print("\n" + df_results.to_string(index=False))
        print("=" * 70)
        
        best_model = max(results, key=lambda x: float(x["Accuracy"].replace("%", "")))
        print(f"\n Top Performing Model: {best_model['Model']} ({best_model['Accuracy']} Accuracy)")
    else:
        print("[-] No models found to evaluate.")

if __name__ == "__main__":
    main()
