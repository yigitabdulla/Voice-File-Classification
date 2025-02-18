import os
import time
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
DATA_PATH = "output"
MAX_LENGTH = 500
N_FEATURES = 8  # Voice + 3 Accel + 3 Gyro + Temp


# 1. Data Loading and Preprocessing
def load_and_process_data(data_path):
    file_list = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    sequences = []
    binary_labels = []
    multiclass_labels = []

    # Create encoders
    fault_encoder = LabelEncoder()
    all_classes = []

    for file in file_list:
        # Extract class from filename
        class_name = file.split('_')[2]
        if 'healthy' in class_name:
            binary_label = 1
            fault_class = 'healthy'
        else:
            binary_label = 0
            fault_class = class_name
        all_classes.append(fault_class)

        # Load CSV
        df = pd.read_csv(os.path.join(data_path, file))
        sensor_data = df[['Voice', 'Acceleration X', 'Acceleration Y', 'Acceleration Z',
                          'Gyro X', 'Gyro Y', 'Gyro Z', 'Temperature']].values

        # Normalize (will finalize after collecting all data)
        sequences.append(sensor_data)
        binary_labels.append(binary_label)
        multiclass_labels.append(fault_class)

    # Pad sequences to fixed length
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, dtype='float32',
                                     padding='post', truncating='post')

    # Normalize data
    scaler = StandardScaler()
    original_shape = padded_sequences.shape
    scaled_data = scaler.fit_transform(padded_sequences.reshape(-1, original_shape[-1]))
    scaled_data = scaled_data.reshape(original_shape)

    # Prepare labels
    binary_labels = np.array(binary_labels)
    fault_encoder.fit([c for c in all_classes if c != 'healthy'])
    multiclass_labels = [fault_encoder.transform([c])[0] if c != 'healthy' else -1
                         for c in multiclass_labels]

    return scaled_data, binary_labels, multiclass_labels, fault_encoder


def create_binary_model():
    model = Sequential([
        Reshape((MAX_LENGTH, N_FEATURES, 1), input_shape=(MAX_LENGTH, N_FEATURES)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),  # Add padding
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),  # Add padding
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_multiclass_model(n_classes):
    model = Sequential([
        Reshape((MAX_LENGTH, N_FEATURES, 1), input_shape=(MAX_LENGTH, N_FEATURES)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),  # Add padding
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),  # Add padding
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 3. Data Preparation
# Load and process all data
X, y_binary, y_multi, fault_encoder = load_and_process_data(DATA_PATH)

# Split data for binary classification
X_train, X_test, y_bin_train, y_bin_test = train_test_split(
    X, y_binary, test_size=0.2, stratify=y_binary, random_state=42
)

# Split fault data for multiclass classification
fault_indices = np.where(y_binary == 0)[0]
X_fault = X[fault_indices]
y_multi_fault = np.array(y_multi)[fault_indices]

X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
    X_fault, y_multi_fault, test_size=0.2, stratify=y_multi_fault, random_state=42
)

# 4. Training

# 5. Hierarchical Prediction Function
def hierarchical_predict(sample):
    # First stage: Healthy vs Fault
    health_prob = binary_model.predict(np.expand_dims(sample, axis=0))[0][0]
    if health_prob > 0.5:
        return "Healthy", None

    # Second stage: Fault classification
    fault_probs = multiclass_model.predict(np.expand_dims(sample, axis=0))[0]
    fault_class = fault_encoder.inverse_transform([np.argmax(fault_probs)])[0]
    return "Fault", fault_class

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score


# 1. Plot Training and Validation Loss
def plot_loss(history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# 2. Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


# 1. Early Stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,  # Stop after 5 epochs without improvement
    restore_best_weights=True  # Restore the best model weights
)


# 2. Function to Print Metrics
def print_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 40)


# 3. Train Binary Model with Early Stopping
binary_model = create_binary_model()
binary_history = binary_model.fit(
    X_train, y_bin_train,
    epochs=50,  # Increase epochs since early stopping will handle overfitting
    batch_size=32,
    validation_data=(X_test, y_bin_test),
    callbacks=[early_stopping]  # Add early stopping
)

# Plot Binary Model Loss
plot_loss(binary_history, "Binary Model Training and Validation Loss")

# 1. Measure Test Time for Binary Model
start_time = time.time()
y_bin_pred = (binary_model.predict(X_test) > 0.5).astype(int)
binary_test_time = time.time() - start_time
print(f"Binary Model Test Time: {binary_test_time:.4f} seconds")

print_metrics(y_bin_test, y_bin_pred, "Binary Model")
plot_confusion_matrix(y_bin_test, y_bin_pred, ["Healthy", "Fault"], "Binary Model Confusion Matrix")

# 4. Train Multiclass Model with Early Stopping
multiclass_model = create_multiclass_model(len(fault_encoder.classes_))
multi_history = multiclass_model.fit(
    X_multi_train, y_multi_train,
    epochs=50,  # Increase epochs since early stopping will handle overfitting
    batch_size=32,
    validation_data=(X_multi_test, y_multi_test),
    callbacks=[early_stopping]  # Add early stopping
)

# Plot Multiclass Model Loss
plot_loss(multi_history, "Multiclass Model Training and Validation Loss")

# 2. Measure Test Time for Multiclass Model
start_time = time.time()
y_multi_pred = multiclass_model.predict(X_multi_test).argmax(axis=1)
multiclass_test_time = time.time() - start_time
print(f"Multiclass Model Test Time: {multiclass_test_time:.4f} seconds")
print_metrics(y_multi_test, y_multi_pred, "Multiclass Model")
plot_confusion_matrix(y_multi_test, y_multi_pred, fault_encoder.classes_, "Multiclass Model Confusion Matrix")


# Save the binary model
binary_model.save('cnn_binary_model.h5')

# Save the multiclass model
multiclass_model.save('cnn_multiclass_model.h5')

# Save the binary model
binary_model.save('cnn_binary_model.h5')

# Save the multiclass model
multiclass_model.save('cnn_multiclass_model.h5')

# Load the saved models
binary_model = load_model('cnn_binary_model.h5')
multiclass_model = load_model('cnn_multiclass_model.h5')


# Iterate through the first 5 test samples
for sample_index in range(5):
    sample = X_test[sample_index]

    # Make hierarchical prediction
    health_status, fault_class = hierarchical_predict(sample)

    # Get real labels
    real_binary_label = y_bin_test[sample_index]
    real_health_status = 'Healthy' if real_binary_label == 1 else 'Fault'

    # Prepare real fault class (if applicable)
    real_fault_class = None
    if real_binary_label == 0:
        real_fault_class = fault_encoder.inverse_transform([y_multi_test[sample_index]])[0]

    # Print results
    print(f"\nSample Index: {sample_index}")
    print("=" * 30)
    print(f"Predicted Health Status: {health_status}")
    if health_status == 'Fault':
        print(f"Predicted Fault Class: {fault_class}")
    print(f"Real Health Status: {real_health_status}")
    if real_health_status == 'Fault':
        print(f"Real Fault Class: {real_fault_class}")
    print("\n" + "-" * 50 + "\n")
