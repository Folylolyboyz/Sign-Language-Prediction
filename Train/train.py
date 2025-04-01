import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import to_onnx
import numpy as np

# Load dataset
df = pd.read_csv("Data Processing/total.csv")

# Prepare features and target
X = df.dropna(axis=0, how='any')
X = X[X.count(axis=1) == 43]
label = X["label"]
X = X.drop(columns=["label"])
labels = "ABCDEFGHIKLMNOPQRSTUVWXY"
d = {j:i for i,j in enumerate(labels)}
y = [d[i] for i in label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# print(X[:1].values.astype(np.float32))

onx = to_onnx(rf_model, X[:1].values.astype(np.float32))
with open("Train/random_forest_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())