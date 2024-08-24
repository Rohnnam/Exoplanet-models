import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess the PS dataset
def preprocess_ps_data(df):
    # Handling NaN values
    numeric_cols = ['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj',
                    'pl_orbeccen', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist']

    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    categorical_cols = ['default_flag', 'discoverymethod', 'soltype', 'pl_controv_flag', 'pl_refname',
                        'pl_orbperlim', 'pl_orbsmaxlim', 'pl_radelim', 'pl_radjlim', 'pl_bmasselim',
                        'pl_bmassjlim', 'pl_orbeccenlim', 'pl_insollim', 'pl_eqtlim', 'ttv_flag',
                        'st_refname', 'st_spectype', 'st_tefflim', 'st_radlim', 'st_masslim', 'st_metlim', 'st_logglim']

    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    for col in numeric_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df

# Function to augment the dataset using weighted combination of features
def augment_data(X, Y, target_size):
    current_size = len(Y)
    samples_needed = target_size - current_size

    weighted_samples = []
    for _ in range(samples_needed):
        idx = np.random.randint(current_size)
        sample = X[idx]
        
        # Random weights
        weight = np.random.uniform(0.1, 0.9)
        new_sample = weight * sample + (1 - weight) * np.random.normal(0, 0.05, sample.shape)  # Adding some noise
        
        weighted_samples.append(new_sample)

    # Convert to numpy array
    weighted_samples = np.array(weighted_samples)

    # Combine original and augmented data
    X_augmented = np.vstack((X, weighted_samples))
    Y_augmented = np.hstack((Y, Y[:len(weighted_samples)]))  # Repeat labels

    return X_augmented, Y_augmented

# Load and preprocess PS dataset
ps_df = pd.read_csv("PS_2024.08.24_01.52.09.csv")
ps_cleaned = preprocess_ps_data(ps_df)

# Features and target variable for PS dataset
X_ps = ps_cleaned[['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj',
                    'pl_orbeccen', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist']]
Y_ps = ps_cleaned['pl_bmasse']  # Adjust based on your target variable

# Ensure balanced dataset with equal positive and negative samples
positive_samples = ps_cleaned[ps_cleaned['pl_bmasse'] > 0].sample(1465, random_state=42)
negative_samples = ps_cleaned[ps_cleaned['pl_bmasse'] <= 0].sample(1465, random_state=42)

# Combine the balanced samples
balanced_ps_df = pd.concat([positive_samples, negative_samples])
X_balanced = balanced_ps_df[['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj',
                              'pl_orbeccen', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist']]
Y_balanced = (balanced_ps_df['pl_bmasse'] > 0).astype(int)  # Convert to binary classification

# Output total, positive, and negative samples for balanced PS dataset
total_samples_ps = len(Y_balanced)
positive_samples_ps = np.sum(Y_balanced == 1)
negative_samples_ps = np.sum(Y_balanced == 0)
print(f'Balanced PS Dataset - Total samples: {total_samples_ps}, Positive samples: {positive_samples_ps}, Negative samples: {negative_samples_ps}')

# Augment the dataset to reach 5000 samples
X_augmented, Y_augmented = augment_data(X_balanced.values, Y_balanced.values, target_size=5000)

# Output total samples after augmentation
print(f'Augmented PS Dataset - Total samples: {len(Y_augmented)}')

# Dimensionality reduction using PCA for PS dataset
pca_ps = PCA(n_components=0.95)  # Retain 95% of variance
X_ps_reduced = pca_ps.fit_transform(X_augmented)

# Train-test split for PS dataset
X_ps_train, X_ps_test, Y_ps_train, Y_ps_test = train_test_split(X_ps_reduced, Y_augmented, test_size=0.2, random_state=42)

# Reshape for LSTM input for PS dataset
X_ps_train = X_ps_train.reshape((X_ps_train.shape[0], X_ps_train.shape[1], 1))
X_ps_test = X_ps_test.reshape((X_ps_test.shape[0], X_ps_test.shape[1], 1))

# Define the model architecture with kernel size of 5 and dropout rate of 0.3
dropout_rate = 0.3
model_ps = Sequential()
model_ps.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=(X_ps_train.shape[1], 1)))
model_ps.add(BatchNormalization())
model_ps.add(Dropout(dropout_rate))  # Dropout layer
model_ps.add(LSTM(128, return_sequences=True))
model_ps.add(BatchNormalization())
model_ps.add(Dropout(dropout_rate))  # Dropout layer
model_ps.add(LSTM(128))
model_ps.add(BatchNormalization())
model_ps.add(Dropout(dropout_rate))  # Dropout layer
model_ps.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model for PS dataset with the best configuration
model_ps.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping and learning rate scheduler for PS dataset
early_stopping_ps = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr_ps = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)

# Train the model for PS dataset
model_ps.fit(X_ps_train, Y_ps_train, epochs=70, validation_data=(X_ps_test, Y_ps_test),
              callbacks=[early_stopping_ps, reduce_lr_ps])

# Evaluate the model for PS dataset
loss_ps, accuracy_ps = model_ps.evaluate(X_ps_test, Y_ps_test)
print(f'PS Dataset - Loss: {loss_ps}, Accuracy: {accuracy_ps}')

## Orbital Period vs. Star Temperature

# Create a visualization for Orbital Period vs. Star Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(data=balanced_ps_df, x='pl_orbper', y='st_teff', hue=Y_balanced, palette='coolwarm', alpha=0.7)
plt.title('Orbital Period vs. Star Temperature')
plt.xlabel('Orbital Period (Days)')
plt.ylabel('Star Temperature (K)')
plt.axhline(y=2.50, color='g', linestyle='--', label='Upper Limit for Habitability (2.50 K)')
plt.axhline(y=-2.50, color='g', linestyle='--', label='Lower Limit for Habitability (-2.50 K)')
plt.axvline(x=-0.006232, color='g', linestyle='--', label='Upper Limit for Habitability (-0.006232 Days)')
plt.axvline(x=0.0, color='g', linestyle='--', label='Lower Limit for Habitability (0.0 Days)')
plt.legend(title='Habitability', loc='upper right', labels=['Not Habitable', 'Habitable'])
plt.grid()
plt.show()


# Calculate planet density
balanced_ps_df['pl_density'] = balanced_ps_df['pl_bmasse'] / (balanced_ps_df['pl_rade']**3)

# Create a histogram for Planet Density Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=balanced_ps_df, x='pl_density', hue=Y_balanced, palette='coolwarm', kde=True, bins=30)
plt.title('Planet Density Distribution')
plt.xlabel('Planet Density (g/cm^3)')
plt.ylabel('Count')
plt.axvline(x=6.125e7, color='g', linestyle='--', label='Upper Limit for Habitability (6.125e7 g/cm^3)')
plt.axvline(x=0.04, color='g', linestyle='--', label='Lower Limit for Habitability (0.04 g/cm^3)')
plt.text(6.125e7 + 1e6, 7600.0, 'Highest Count at -5.278e7 g/cm^3', rotation=90)
plt.text(1.80e9 + 1e7, 248.0, 'Highest Count at -5.26e7 g/cm^3', rotation=90)
plt.legend(title='Habitability', loc='upper right', labels=['Not Habitable', 'Habitable'])
plt.grid()
plt.show()



# Create a scatter plot for Star Mass vs. Planet Mass
plt.figure(figsize=(10, 6))
sns.scatterplot(data=balanced_ps_df, x='st_mass', y='pl_bmasse', hue=Y_balanced, palette='coolwarm', alpha=0.7)
plt.title('Star Mass vs. Planet Mass')
plt.xlabel('Star Mass (Solar Masses)')
plt.ylabel('Planet Mass (Jupiter Masses)')
plt.xscale('log')
plt.yscale('log')
plt.axhline(y=0.3, color='r', linestyle='--', label='Approx. Lower Limit for Habitability')
plt.legend(title='Habitability', loc='upper left', labels=['Not Habitable', 'Habitable'])
plt.grid()
plt.show()


