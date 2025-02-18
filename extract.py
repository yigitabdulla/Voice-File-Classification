import os
import pandas as pd
import numpy as np
from scipy.signal import stft


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
df.to_csv('merged_features.csv', index=False)

