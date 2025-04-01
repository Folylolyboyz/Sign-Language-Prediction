import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.optimizers.experimental import AdamW
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tf2onnx
import onnx

# Load dataset
file_path = "Data Processing/total.csv"
df = pd.read_csv(file_path)

# Drop missing values
df_clean = df.dropna()

# Encode labels
label_encoder = LabelEncoder()
df_clean["label"] = label_encoder.fit_transform(df_clean["label"])

# Split features and labels
X = df_clean.drop(columns=["label"])
y = df_clean["label"].values

# Define parameters
num_trees = 40  # Number of "trees" in the ensemble
input_shape = (X.shape[1],)
num_classes = len(np.unique(y))

# Create a better tree (subnetwork)
def create_better_tree(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # First dense layer
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # DenseNet-inspired Residual Block
    x_res = layers.Dense(512, activation="relu")(x)
    x_res = layers.BatchNormalization()(x_res)
    x_res = layers.Dropout(0.4)(x_res)

    # Align shapes before adding
    x_res = layers.Dense(512)(x_res)  # Projection to match shapes
    x = layers.Add()([x, x_res])  # Residual connection

    # Another Residual Block with more depth
    x_res = layers.Dense(256, activation="relu")(x)
    x_res = layers.BatchNormalization()(x_res)
    x_res = layers.Dropout(0.4)(x_res)

    # Align shapes again before adding
    x_res = layers.Dense(512)(x_res)  # Projection to match shapes
    x = layers.Add()([x, x_res])  # Residual connection

    # Additional layers to increase depth
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

# Create ensemble of models (trees)
trees = [create_better_tree(input_shape, num_classes) for _ in range(num_trees)]

# Define ensemble output (average of all tree outputs)
inputs = keras.Input(shape=input_shape)
outputs = layers.Average()([tree(inputs) for tree in trees])

# Final ensemble model
ensemble_model = keras.Model(inputs, outputs)
ensemble_model.compile(optimizer=AdamW(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
ensemble_model.fit(X, y, epochs=20, batch_size=16, validation_split=0.2)

# Save the Keras model
ensemble_model.save("Train/asl_ensemble_model.keras")

# Convert and save as ONNX
onnx_model_path = "Train/asl_ensemble_model.onnx"
spec = (tf.TensorSpec((None, input_shape[0]), tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(ensemble_model, input_signature=spec)

# Save the ONNX model using the onnx library
onnx.save_model(onnx_model, onnx_model_path)
