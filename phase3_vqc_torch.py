# ...existing code...
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Reproducibility
torch.manual_seed(42)

# -------------------------------
# Load Data
# -------------------------------
X = pd.read_csv("data/X_scaled.csv").values
y = pd.read_csv("data/y.csv").values.ravel()

# initial train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# further split train -> train + val for validation monitoring
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# -------------------------------
# Quantum Circuit Setup
# -------------------------------
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

# -------------------------------
# Hybrid Classical + Quantum Model
# -------------------------------
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        # small classical preprocessing -> expand then map to n_qubits (quantum input size must match)
        self.pre_fc = nn.Linear(n_qubits, 64)
        self.fc1 = nn.Linear(64, n_qubits)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        # quantum layer
        self.quantum = QuantumLayer()
        # post-quantum classical layers
        self.fc2 = nn.Linear(n_qubits, 32)
        self.fc3 = nn.Linear(32, 1)
        # NOTE: removed final sigmoid because we'll use BCEWithLogitsLoss for stability
    def forward(self, x):
        x = self.pre_fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.quantum(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)  # raw logit output
        return x

model = HybridModel()
# use BCEWithLogitsLoss and keep model output as logits
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
# removed unexpected 'verbose' kwarg
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# -------------------------------
# Training Loop (25 epochs)
# -------------------------------
epochs = 25
best_val_acc = 0.0
best_state = None

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)                  # logits
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        probs = torch.sigmoid(outputs)       # convert to probabilities for metrics
        preds = (probs >= 0.5).float()
        correct += preds.eq(yb).sum().item()
        total += xb.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    # validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            outputs = model(xb)  # logits
            l = criterion(outputs, yb)
            val_loss += l.item() * xb.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            val_correct += preds.eq(yb).sum().item()
            val_total += xb.size(0)
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()

    if epoch % 1 == 0:
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

# Save best model (or final if none improved)
if best_state is not None:
    torch.save(best_state, "models/hybrid_vqc_model_best.pth")
else:
    torch.save(model.state_dict(), "models/hybrid_vqc_model.pth")

print(f"Phase 3 complete: Best Val Acc: {best_val_acc:.4f} - Model saved.")
# ...existing code...