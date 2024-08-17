import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path, flux_col, sequence_length=100):
    data = pd.read_csv(file_path)
    flux = data[flux_col].values
    
    # Normalize flux values
    scaler = MinMaxScaler()
    flux_normalized = scaler.fit_transform(flux.reshape(-1, 1))
    
    # Create sequences
    sequences = []
    for i in range(len(flux_normalized) - sequence_length + 1):
        seq = flux_normalized[i:i+sequence_length]
        sequences.append(seq)
    
    return np.array(sequences)


# Load datasets
sequence_length = 100
tres_flux = load_and_preprocess_data('TrES_2024.08.17_00.58.37.csv', 'rmag', sequence_length)
cluster_flux = load_and_preprocess_data('Cluster_2024.08.17_00.58.25.csv', 'rmag', sequence_length)
xo_flux = load_and_preprocess_data('XO_2024.08.17_00.59.33.csv', 'jmag', sequence_length)
kelt_flux = load_and_preprocess_data('KELTP_2024.08.17_00.58.08.csv', 'rmag', sequence_length)

# Print information about the loaded data
print(f"TrES samples: {tres_flux.shape[0]}")
print(f"Cluster samples: {cluster_flux.shape[0]}")
print(f"XO samples: {xo_flux.shape[0]}")
print(f"KELT samples: {kelt_flux.shape[0]}")

# Combine all datasets
all_flux = np.vstack([tres_flux, cluster_flux, xo_flux, kelt_flux])

# Create labels (1 for exoplanets, 0 for non-exoplanets)
labels = np.concatenate([
    np.ones(tres_flux.shape[0]),
    np.zeros(cluster_flux.shape[0]),
    np.ones(xo_flux.shape[0]),
    np.zeros(kelt_flux.shape[0])
])


print(f"Total samples: {all_flux.shape[0]}")
print(f"Positive samples: {np.sum(labels)}")
print(f"Negative samples: {len(labels) - np.sum(labels)}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_flux, labels, test_size=0.2, random_state=42, stratify=labels)

# Build the model
def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Train the model
model = build_model(input_shape=(sequence_length, 1))
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    class_weight={0: 1., 1: 7.}  # Adjust class weights if needed
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

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
