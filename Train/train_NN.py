import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from tensorflow.keras.optimizers.experimental import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
import tf2onnx
import onnx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

file_path = "Data Processing/total.csv"
df = pd.read_csv(file_path)

df_clean = df.dropna()

labels = "ABCDEFGHIKLMNOPQRSTUVWXY"
d = {j: i for i, j in enumerate(labels)}
df_clean["label"] = df_clean["label"].map(d)

X = df_clean.drop(columns=["label"])
y = pd.get_dummies(df_clean["label"]).values  # One-hot encoding


X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

input_shape = (X.shape[1],)
num_classes = y.shape[1]

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

def build_optimized_model():
    inputs = keras.Input(shape=(42,))

    # Input Block
    x = layers.Dense(512, activation=None)(inputs)  # No activation yet
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    # Residual Block 1
    res1 = layers.Dense(128, activation=None)(x)
    res1 = layers.BatchNormalization()(res1)
    res1 = layers.LeakyReLU()(res1)
    res1 = layers.Dense(128, activation=None)(res1)
    res1 = layers.BatchNormalization()(res1)
    res1 = layers.LeakyReLU()(res1)

    x = layers.Dense(128, activation=None)(x)  # Projection layer
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, res1])  # Skip connection
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    # Residual Block 2
    res2 = layers.Dense(64, activation=None)(x)
    res2 = layers.BatchNormalization()(res2)
    res2 = layers.LeakyReLU()(res2)
    res2 = layers.Dense(64, activation=None)(res2)
    res2 = layers.BatchNormalization()(res2)
    res2 = layers.LeakyReLU()(res2)

    x = layers.Dense(64, activation=None)(x)  # Projection layer
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, res2])  # Skip connection
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    # Final Layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Output Layer
    outputs = layers.Dense(24, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Train Models
num_models = 5
models = [build_optimized_model() for _ in range(num_models)]

for i, model in enumerate(models):
    print(f"Training model {i+1}/{num_models}...")
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

# Evaluate models on test set
test_accuracies = [model.evaluate(X_test, y_test, verbose=0)[1] for model in models]
avg_test_accuracy = np.mean(test_accuracies)
print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")

# Ensemble predictions
def ensemble_predict(models, X):
    predictions = np.array([model.predict(X) for model in models])
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction

y_pred = ensemble_predict(models, X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
inverse_d = {v: k for k, v in d.items()}  # Reverse mapping
y_pred_labels = [inverse_d[i] for i in y_pred_classes]
# for i, label in enumerate(y_pred_labels[:10]):  # Show first 10 predictions
#     print(f"Sample {i+1}: Predicted Label - {label}")

# Generate confusion matrix
true_classes = np.argmax(y_test, axis=1)
cm = confusion_matrix(true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Convert and save model as ONNX
def build_ensemble_model(models, input_shape=(42,)):
    inputs = keras.Input(shape=input_shape)
    predictions = [model(inputs) for model in models]
    avg_prediction = layers.Average()(predictions)
    ensemble_model = Model(inputs=inputs, outputs=avg_prediction)
    return ensemble_model

ensemble_model = build_ensemble_model(models)

onnx_model_path = "Train/ensemble_model.onnx"
spec = (tf.TensorSpec((None, 42), tf.float32),)
model_proto, _ = tf2onnx.convert.from_keras(ensemble_model, input_signature=spec, opset=13)
onnx.save_model(model_proto, onnx_model_path)
print(f"Model saved as {onnx_model_path}")
