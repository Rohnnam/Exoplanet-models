import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1_l2
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path, flux_cols):
    data = pd.read_csv(file_path)
    flux_data = data[flux_cols].astype(float).values
    imputer = SimpleImputer(strategy='median')
    flux_data_imputed = imputer.fit_transform(flux_data)
    scaler = StandardScaler()
    flux_normalized = scaler.fit_transform(flux_data_imputed)
    return flux_normalized

def adjust_features(data, max_features):
    current_features = data.shape[1]
    if current_features < max_features:
        # Pad with zeros
        padding = np.zeros((data.shape[0], max_features - current_features))
        return np.concatenate((data, padding), axis=1)
    elif current_features > max_features:
        # Trim to max features
        return data[:, :max_features]
    return data

# Load datasets
tres_flux = load_and_preprocess_data("TrES_2024.08.17_00.58.37.csv", ['rmag', 'bmag', 'vmag', 'ra', 'dec'])
cluster_flux = load_and_preprocess_data("Cluster_2024.08.17_00.58.25.csv", ['rmag', 'bmag', 'vmag', 'ra', 'dec'])
xo_flux = load_and_preprocess_data("XO_2024.08.17_00.59.33.csv", ['jmag', 'hmag', 'ra', 'dec'])
kelt_flux = load_and_preprocess_data("KELTP_2024.08.17_00.58.08.csv", ['rmag', 'dec', 'starthjd', 'endhjd'])

# Adjust features
max_features = max(tres_flux.shape[1], cluster_flux.shape[1], xo_flux.shape[1], kelt_flux.shape[1])
tres_flux = adjust_features(tres_flux, max_features)
cluster_flux = adjust_features(cluster_flux, max_features)
xo_flux = adjust_features(xo_flux, max_features)
kelt_flux = adjust_features(kelt_flux, max_features)

# Combine datasets
all_flux = np.vstack([tres_flux, cluster_flux, xo_flux, kelt_flux])
labels = np.concatenate([
    np.ones(tres_flux.shape[0]),
    np.zeros(cluster_flux.shape[0]),
    np.ones(xo_flux.shape[0]),
    np.zeros(kelt_flux.shape[0])
])

def light_curve(exoplanet_position, star_position, num_points=100):
    # Generate angles for light rays
    angles = np.linspace(0, 2 * np.pi, num_points)
    
    # Star coordinates
    star_x, star_y = star_position
    
    # Exoplanet coordinates
    exoplanet_x, exoplanet_y = exoplanet_position
    
    # Calculate light rays
    light_x = star_x + np.cos(angles)
    light_y = star_y + np.sin(angles)
    
    # Calculate the bending effect (simple model)
    bending_factor = 0.5  # Adjust this for more or less bending
    light_y_bent = light_y - bending_factor * (light_x - exoplanet_x)**2
    
    return light_x, light_y_bent

# Downsample negative samples to match positive samples
num_positive_samples = 28467
positive_indices = np.where(labels == 1)[0]
negative_indices = np.where(labels == 0)[0]
if len(negative_indices) > num_positive_samples:
    negative_indices = np.random.choice(negative_indices, num_positive_samples, replace=False)
balanced_indices = np.concatenate([positive_indices, negative_indices])
all_flux_balanced = all_flux[balanced_indices]
labels_balanced = labels[balanced_indices]

# Print sample statistics
print(f"Total samples: {all_flux_balanced.shape[0]}")
print(f"Positive samples: {np.sum(labels_balanced)}")
print(f"Negative samples: {len(labels_balanced) - np.sum(labels_balanced)}")

# PCA
pca = PCA(n_components=0.95)
all_flux_reduced = pca.fit_transform(all_flux_balanced)

# Split the data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(all_flux_reduced, labels_balanced, test_size=0.2, random_state=42, stratify=labels_balanced)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)  # 0.25 of 0.8 = 0.2

# Reshape input for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define class weights
class_weight = {0: 1.0, 1: (len(labels_balanced) / (2 * np.sum(labels_balanced)))}

# Build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape, kernel_regularizer=l1_l2(l1=0.005, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(LSTM(128, return_sequences=False, kernel_regularizer=l1_l2(l1=0.005, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Define early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
model = build_model(input_shape=(X_train.shape[1], 1))
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),  # Use validation data
    shuffle=True,
    callbacks=[early_stopping, lr_reduction],
    class_weight=class_weight
)

# Final evaluation of the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Visualization of light curving around an exoplanet
star_position = (0, 0)
exoplanet_position = (1, 0)

# Generate light curve
light_x, light_y = light_curve(exoplanet_position, star_position)

# Set up the plot
plt.figure(figsize=(10, 8))
plt.style.use('seaborn-dark')  # Use a dark background for space effect

# Plot light rays with a gradient effect
for i in range(len(light_x) - 1):
    plt.plot(light_x[i:i+2], light_y[i:i+2], color=(1.0, 1.0 - i/len(light_x), 0), lw=2)  # Gradient from yellow to blue

# Plot the star with a glow effect
plt.scatter(*star_position, color='yellow', s=300, edgecolor='orange', linewidth=2, label='Star')

# Plot the exoplanet with surface features
exoplanet_circle = plt.Circle(exoplanet_position, 0.1, color='red', alpha=0.8, label='Exoplanet')
plt.gca().add_artist(exoplanet_circle)

# Add shadows to the exoplanet for depth
shadow_circle = plt.Circle(exoplanet_position, 0.1, color='black', alpha=0.3)
plt.gca().add_artist(shadow_circle)

# Set limits and labels
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axhline(0, color='white', lw=0.5, ls='--')
plt.axvline(0, color='white', lw=0.5, ls='--')
plt.title('Light Curving Around an Exoplanet', fontsize=16)
plt.xlabel('X Position', fontsize=14)
plt.ylabel('Y Position', fontsize=14)
plt.legend()
plt.grid(False)
plt.gca().set_aspect('equal', adjustable='box')

# Show the plot
plt.show()

