# phase1_data_prep.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load dataset
df = pd.read_csv("data/heart.csv")

# 2️⃣ Inspect dataset
print("First 5 rows:\n", df.head())
print("\nDataset info:\n")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

# 3️⃣ Handle missing values (if any)
# Currently, your dataset has no missing values.
# If there were missing values, you could uncomment below:
# df = df.dropna()
# df.fillna(df.mean(), inplace=True)

# 4️⃣ Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 5️⃣ Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6️⃣ Save preprocessed dataset (optional)
pd.DataFrame(X_scaled, columns=X.columns).to_csv("data/X_scaled.csv", index=False)
pd.DataFrame(y, columns=['target']).to_csv("data/y.csv", index=False)

print("\n✅ Phase 1 completed. Preprocessed data saved as:")
print(" - data/X_scaled.csv")
print(" - data/y.csv")
