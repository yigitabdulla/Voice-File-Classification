import os
import time

import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import stft
from sklearn import clone
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, \
    classification_report
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def extract_and_merge_features(input_folder):
    # List to hold dataframes
    dataframes = []

    # Parameters for STFT
    fs = 90  # Updated sample rate to 90 Hz
    nperseg = int(2 * fs)  # Common practice: twice the sampling frequency
    noverlap = int(nperseg * 0.75)  # 75% overlap is a common choice for balancing resolution

    # Columns to transform
    columns_to_transform = ['Speed', 'Voice', 'Acceleration X', 'Acceleration Y', 'Acceleration Z', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Temperature']

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            # Construct full file path
            file_path = os.path.join(input_folder, filename)
            # Read the CSV file
            data = pd.read_csv(file_path)
            # Extract the class value from the filename
            class_value = filename.split('_')[2]
            # If the class value starts with 'healthy', return 'healthy'
            if class_value.startswith('healthy'):
                class_value = 'healthy'

            # Dictionary to hold features
            stft_features = {}

            # Compute STFT and extract features for each column across the entire signal
            for col in columns_to_transform:
                if col in data.columns:
                    # Perform STFT
                    f, t, Zxx = stft(data[col].values, fs=fs, nperseg=nperseg, noverlap=noverlap)

                    # Feature extraction across entire signal: mean, variance, energy, median, max, min, skewness, kurtosis, and entropy of magnitude spectrum
                    magnitude_spectrum = np.abs(Zxx)
                    mean_mag = np.mean(magnitude_spectrum)
                    var_mag = np.var(magnitude_spectrum)
                    energy_mag = np.sum(magnitude_spectrum**2)
                    median_mag = np.median(magnitude_spectrum)
                    max_mag = np.max(magnitude_spectrum)
                    min_mag = np.min(magnitude_spectrum)
                    skew_mag = pd.Series(magnitude_spectrum.flatten()).skew()
                    kurt_mag = pd.Series(magnitude_spectrum.flatten()).kurtosis()
                    entropy_mag = -np.sum(magnitude_spectrum * np.log(magnitude_spectrum + 1e-12))

                    # Store features in the dictionary
                    stft_features[f'{col}_mean_mag'] = [mean_mag]
                    stft_features[f'{col}_var_mag'] = [var_mag]
                    stft_features[f'{col}_energy_mag'] = [energy_mag]
                    stft_features[f'{col}_median_mag'] = [median_mag]
                    stft_features[f'{col}_max_mag'] = [max_mag]
                    stft_features[f'{col}_min_mag'] = [min_mag]
                    stft_features[f'{col}_skew_mag'] = [skew_mag]
                    stft_features[f'{col}_kurt_mag'] = [kurt_mag]
                    stft_features[f'{col}_entropy_mag'] = [entropy_mag]

            # Add the 'class' column and fill it with the determined value
            stft_features['class'] = [class_value]
            # Convert to a DataFrame with column names and a single row
            features_df = pd.DataFrame(stft_features)
            # Append the dataframe to the list
            dataframes.append(features_df)

    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df

df = extract_and_merge_features('output')

# Load data
df = extract_and_merge_features('output')
X = df.drop(columns=['class'])
y = df['class']
y_binary = y.apply(lambda x: 1 if x == 'healthy' else 0)

# Split data into train/test (stratified by binary labels)
X_train, X_test, y_train_bin, y_test_bin = train_test_split(
    X,
    y_binary,
    test_size=0.2,
    stratify=y_binary,
    random_state=42
)

# Initialize models for both stages
binary_models = [
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(random_state=42))
]

multi_class_models = [
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(random_state=42))
]


# Function to evaluate models with stratified CV
def evaluate_models(X, y, models, stage_name):
    best_model = None
    best_score = 0
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for name, model in models:
        print(f"\nEvaluating {name} for {stage_name}...")
        y_pred = cross_val_predict(clone(model), X, y, cv=skf)

        # Print metrics
        print(classification_report(y, y_pred))
        print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
        print(f"Precision: {precision_score(y, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(y, y_pred, average='weighted'):.4f}")
        print(f"F1 Score: {f1_score(y, y_pred, average='weighted'):.4f}")

        # Plot confusion matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=(stage_name == 'binary') and ['Healthy', 'Unhealthy'] or np.unique(y),
                    yticklabels=(stage_name == 'binary') and ['Healthy', 'Unhealthy'] or np.unique(y))
        plt.title(f'{name} - {stage_name} Classification')
        plt.show()

        # Track best model
        current_score = f1_score(y, y_pred, average='weighted')
        if current_score > best_score:
            best_score = current_score
            best_model = (name, model)

    return best_model


# Binary Classification Evaluation (using only training data)
print("=== Binary Classification Evaluation ===")
best_binary = evaluate_models(X_train, y_train_bin, binary_models, 'binary')

# Final evaluation on test set with confusion matrix
best_model = clone(best_binary[1]).fit(X_train, y_train_bin)
y_pred_test = best_model.predict(X_test)
print("\nTest Set Performance:")
print(classification_report(y_test_bin, y_pred_test))

# Binary confusion matrix
cm_binary = confusion_matrix(y_test_bin, y_pred_test)
plt.figure(figsize=(6,4))
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Unhealthy', 'Healthy'],
           yticklabels=['Unhealthy', 'Healthy'])
plt.title(f'{best_binary[0]} - Binary Test Set Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Split unhealthy data (using original indices)
unhealthy_mask = y_binary == 0
X_unhealthy = X[unhealthy_mask]
y_unhealthy_multi = y[unhealthy_mask]

# Split multi-class data
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
    X_unhealthy,
    y_unhealthy_multi,
    test_size=0.2,
    stratify=y_unhealthy_multi,
    random_state=42
)

# Multi-class Evaluation (using training portion)
print("\n=== Multi-class Classification Evaluation ===")
best_multi = evaluate_models(X_multi_train, y_multi_train, multi_class_models, 'multi-class')

# Final evaluation on test set with confusion matrix
best_multi_model = clone(best_multi[1]).fit(X_multi_train, y_multi_train)
y_multi_pred_test = best_multi_model.predict(X_multi_test)

print("\nTest Set Performance:")
print(classification_report(y_multi_test, y_multi_pred_test))

# Multi-class confusion matrix
cm_multi = confusion_matrix(y_multi_test, y_multi_pred_test)
class_labels = np.unique(np.concatenate([y_multi_test, y_multi_pred_test]))
plt.figure(figsize=(10,8))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
           xticklabels=class_labels,
           yticklabels=class_labels)
plt.title(f'{best_multi[0]} - Multi-class Test Set Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Retrain final models on full training data
final_binary_model = best_binary[1].fit(X_train, y_train_bin)
final_multi_model = best_multi[1].fit(X_multi_train, y_multi_train)

# Save models
joblib.dump(final_binary_model, 'best_machine_learning_binary_model.pkl')
joblib.dump(final_multi_model, 'best__machine_learning_multi_model.pkl')

# Binary Classification Test Evaluation for All Models
print("\n=== Binary Models Test Performance ===")
binary_test_times = []

for name, model in binary_models:
    print(f"\nEvaluating {name} on Binary Test Set...")

    # Train model
    clf = clone(model).fit(X_train, y_train_bin)

    # Measure prediction time
    start_time = time.time()
    y_pred = clf.predict(X_test)
    test_time = time.time() - start_time
    binary_test_times.append((name, test_time))

    # Metrics
    print(f"Test Prediction Time: {test_time:.4f} seconds")
    print(classification_report(y_test_bin, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test_bin, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unhealthy', 'Healthy'],
                yticklabels=['Unhealthy', 'Healthy'])
    plt.title(f'{name} - Binary Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Multi-class Classification Test Evaluation for All Models
print("\n=== Multi-class Models Test Performance ===")
multi_test_times = []

for name, model in multi_class_models:
    print(f"\nEvaluating {name} on Multi-class Test Set...")

    # Train model
    clf = clone(model).fit(X_multi_train, y_multi_train)

    # Measure prediction time
    start_time = time.time()
    y_pred = clf.predict(X_multi_test)
    test_time = time.time() - start_time
    multi_test_times.append((name, test_time))

    # Metrics
    print(f"Test Prediction Time: {test_time:.4f} seconds")
    print(classification_report(y_multi_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_multi_test, y_pred)
    class_labels = np.unique(np.concatenate([y_multi_test, y_pred]))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f'{name} - Multi-class Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Display timing results
print("\nBinary Model Prediction Times:")
for name, t in binary_test_times:
    print(f"{name}: {t:.4f} seconds")

print("\nMulti-class Model Prediction Times:")
for name, t in multi_test_times:
    print(f"{name}: {t:.4f} seconds")


# Load the saved models
binary_model = joblib.load('best_machine_learning_binary_model.pkl')
multi_model = joblib.load('best__machine_learning_multi_model.pkl')

# Select the first 5 test samples
test_samples = X_test.iloc[10:20]  # Using original test set from binary classification

# Predict and display results
print("\n=== Predictions on First 5 Test Samples ===")
for i in range(len(test_samples)):
    # Get sample and its actual class
    sample = test_samples.iloc[i:i + 1]  # Maintain DataFrame structure for prediction
    actual_class = df.loc[test_samples.index[i], 'class']  # Get true class from original data

    # Binary prediction
    binary_pred = binary_model.predict(sample)[0]

    # Multi-class prediction if needed
    if binary_pred == 1:
        predicted_class = 'healthy'
    else:
        predicted_class = multi_model.predict(sample)[0]

    # Print results
    print(f"\nSample {i + 1}:")
    print(f"Predicted: {predicted_class}")
    print(f"Actual:    {actual_class}")