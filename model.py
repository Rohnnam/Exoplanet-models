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
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from multiprocessing import freeze_support

def load_and_preprocess_data(file_path, flux_cols):
    try:
        data = pd.read_csv(file_path)
        flux_data = data[flux_cols].astype(float).values
        imputer = SimpleImputer(strategy='median')
        flux_data_imputed = imputer.fit_transform(flux_data)
        scaler = StandardScaler()
        flux_normalized = scaler.fit_transform(flux_data_imputed)
        return flux_normalized
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def adjust_features(data, max_features):
    current_features = data.shape[1]
    if current_features < max_features:
        padding = np.zeros((data.shape[0], max_features - current_features))
        return np.concatenate((data, padding), axis=1)
    elif current_features > max_features:
        return data[:, :max_features]
    return data

def parallel_load_and_preprocess(args):
    file_path, flux_cols, max_features = args
    data = load_and_preprocess_data(file_path, flux_cols)
    if data is not None:
        return adjust_features(data, max_features)
    return None

def light_curve(exoplanet_position, star_position, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    star_x, star_y = star_position
    exoplanet_x, exoplanet_y = exoplanet_position
    light_x = star_x + np.cos(angles)
    light_y = star_y + np.sin(angles)
    bending_factor = 0.5
    light_y_bent = light_y - bending_factor * (light_x - exoplanet_x)**2
    return light_x, light_y_bent

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

def main():
    file_paths = [
        "TrES_2024.08.17_00.58.37.csv",
        "Cluster_2024.08.17_00.58.25.csv",
        "XO_2024.08.17_00.59.33.csv",
        "KELTP_2024.08.17_00.58.08.csv"
    ]

    flux_cols_list = [
        ['rmag', 'bmag', 'vmag', 'ra', 'dec'],
        ['rmag', 'bmag', 'vmag', 'ra', 'dec'],
        ['jmag', 'hmag', 'ra', 'dec'],
        ['rmag', 'dec', 'starthjd', 'endhjd']
    ]

    max_features = max(len(cols) for cols in flux_cols_list)

    with ProcessPoolExecutor() as executor:
        all_flux_list = list(executor.map(parallel_load_and_preprocess, 
                                          zip(file_paths, flux_cols_list, [max_features]*len(file_paths))))

    all_flux_list = [flux for flux in all_flux_list if flux is not None]

    if len(all_flux_list) < len(file_paths):
        print("Some datasets failed to load. Exiting.")
        return

    tres_flux, cluster_flux, xo_flux, kelt_flux = all_flux_list

    all_flux = np.vstack([tres_flux, cluster_flux, xo_flux, kelt_flux])
    labels = np.concatenate([
        np.ones(tres_flux.shape[0]),
        np.zeros(cluster_flux.shape[0]),
        np.ones(xo_flux.shape[0]),
        np.zeros(kelt_flux.shape[0])
    ])

    num_positive_samples = 28467
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    if len(negative_indices) > num_positive_samples:
        negative_indices = np.random.choice(negative_indices, num_positive_samples, replace=False)
    balanced_indices = np.concatenate([positive_indices, negative_indices])
    all_flux_balanced = all_flux[balanced_indices]
    labels_balanced = labels[balanced_indices]

    print(f"Total samples: {all_flux_balanced.shape[0]}")
    print(f"Positive samples: {np.sum(labels_balanced)}")
    print(f"Negative samples: {len(labels_balanced) - np.sum(labels_balanced)}")

    pca = PCA(n_components=0.95)
    all_flux_reduced = pca.fit_transform(all_flux_balanced)

    X_temp, X_test, y_temp, y_test = train_test_split(all_flux_reduced, labels_balanced, test_size=0.2, random_state=42, stratify=labels_balanced)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    model = build_model(input_shape=(X_train.shape[1], 1))
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        shuffle=True,
        callbacks=[early_stopping, lr_reduction],
        class_weight=class_weight_dict
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

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
    
    star_position = (0, 0)
    exoplanet_position = (1, 0)

    light_x, light_y = light_curve(exoplanet_position, star_position)

    plt.figure(figsize=(10, 8), facecolor='black')
    plt.style.use('dark_background')

    plt.gca().set_facecolor('#000033')

    for i in range(len(light_x) - 1):
        plt.plot(light_x[i:i+2], light_y[i:i+2], color=(1.0, 1.0 - i/len(light_x), i/len(light_x)), lw=2, alpha=0.7)

    star_glow = plt.Circle(star_position, 0.15, color='yellow', alpha=0.3)
    plt.gca().add_artist(star_glow)
    plt.scatter(*star_position, color='yellow', s=300, edgecolor='orange', linewidth=2, label='Star', zorder=3)

    exoplanet_circle = plt.Circle(exoplanet_position, 0.1, color='#8B4513', alpha=0.8, label='Exoplanet')
    plt.gca().add_artist(exoplanet_circle)

    shadow_circle = plt.Circle(exoplanet_position, 0.1, color='black', alpha=0.3)
    plt.gca().add_artist(shadow_circle)
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color='white', lw=0.5, ls='--', alpha=0.3)
    plt.axvline(0, color='white', lw=0.5, ls='--', alpha=0.3)
    plt.title('Light Curving Around an Exoplanet', fontsize=16, color='white')
    plt.xlabel('X Position', fontsize=14, color='white')
    plt.ylabel('Y Position', fontsize=14, color='white')
    plt.legend(facecolor='#000033', edgecolor='white')
    plt.grid(False)
    plt.gca().set_aspect('equal', adjustable='box')

    for _ in range(100):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        size = np.random.uniform(0.1, 2)
        alpha = np.random.uniform(0.1, 1)
        plt.scatter(x, y, s=size, c='white', alpha=alpha)

    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()
