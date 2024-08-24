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

def preprocess_ps_data(df):
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


def augment_data(X, Y, target_size):
    current_size = len(Y)
    samples_needed = target_size - current_size

    weighted_samples = []
    for _ in range(samples_needed):
        idx = np.random.randint(current_size)
        sample = X[idx]
        

        weight = np.random.uniform(0.1, 0.9)
        new_sample = weight * sample + (1 - weight) * np.random.normal(0, 0.05, sample.shape) 
        
        weighted_samples.append(new_sample)

    weighted_samples = np.array(weighted_samples)
    X_augmented = np.vstack((X, weighted_samples))
    Y_augmented = np.hstack((Y, Y[:len(weighted_samples)]))

    return X_augmented, Y_augmented

ps_df = pd.read_csv("PS_2024.08.24_01.52.09.csv")
ps_cleaned = preprocess_ps_data(ps_df)

X_ps = ps_cleaned[['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj',
                    'pl_orbeccen', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist']]
Y_ps = ps_cleaned['pl_bmasse'] 

positive_samples = ps_cleaned[ps_cleaned['pl_bmasse'] > 0].sample(1465, random_state=42)
negative_samples = ps_cleaned[ps_cleaned['pl_bmasse'] <= 0].sample(1465, random_state=42)

balanced_ps_df = pd.concat([positive_samples, negative_samples])
X_balanced = balanced_ps_df[['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj',
                              'pl_orbeccen', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist']]
Y_balanced = (balanced_ps_df['pl_bmasse'] > 0).astype(int) 

total_samples_ps = len(Y_balanced)
positive_samples_ps = np.sum(Y_balanced == 1)
negative_samples_ps = np.sum(Y_balanced == 0)
print(f'Balanced PS Dataset - Total samples: {total_samples_ps}, Positive samples: {positive_samples_ps}, Negative samples: {negative_samples_ps}')

X_augmented, Y_augmented = augment_data(X_balanced.values, Y_balanced.values, target_size=5000)

print(f'Augmented PS Dataset - Total samples: {len(Y_augmented)}')

pca_ps = PCA(n_components=0.95) 
X_ps_reduced = pca_ps.fit_transform(X_augmented)

X_ps_train, X_ps_test, Y_ps_train, Y_ps_test = train_test_split(X_ps_reduced, Y_augmented, test_size=0.2, random_state=42)

X_ps_train = X_ps_train.reshape((X_ps_train.shape[0], X_ps_train.shape[1], 1))
X_ps_test = X_ps_test.reshape((X_ps_test.shape[0], X_ps_test.shape[1], 1))

dropout_rate = 0.3
model_ps = Sequential()
model_ps.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=(X_ps_train.shape[1], 1)))
model_ps.add(BatchNormalization())
model_ps.add(Dropout(dropout_rate))
model_ps.add(LSTM(128, return_sequences=True))
model_ps.add(BatchNormalization())
model_ps.add(Dropout(dropout_rate)) 
model_ps.add(LSTM(128))
model_ps.add(BatchNormalization())
model_ps.add(Dropout(dropout_rate)) 
model_ps.add(Dense(1, activation='sigmoid'))  

model_ps.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


early_stopping_ps = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr_ps = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)

model_ps.fit(X_ps_train, Y_ps_train, epochs=70, validation_data=(X_ps_test, Y_ps_test),
              callbacks=[early_stopping_ps, reduce_lr_ps])
loss_ps, accuracy_ps = model_ps.evaluate(X_ps_test, Y_ps_test)
print(f'PS Dataset - Loss: {loss_ps}, Accuracy: {accuracy_ps}')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=balanced_ps_df, x='pl_eqt', y='pl_rade', hue=Y_balanced, palette='coolwarm', alpha=0.7)
plt.title('Exoplanet Habitability Visualization')
plt.xlabel('Equilibrium Temperature (K)')
plt.ylabel('Planet Radius (Earth Radii)')
plt.axhline(y=1, color='r', linestyle='--', label='Earth Radius')
plt.axvline(x=273, color='g', linestyle='--', label='Temperature for Liquid Water')
plt.legend(title='Habitability', loc='upper right', labels=['Not Habitable', 'Habitable'])
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=balanced_ps_df, x='st_mass', y='pl_bmasse', hue=Y_balanced, palette='coolwarm', alpha=0.7)
plt.title('Star Mass vs. Planet Mass')
plt.xlabel('Star Mass (Solar Masses)')
plt.ylabel('Planet Mass (Earth Masses)')
plt.yscale('log')
plt.legend(title='Habitability', loc='upper right', labels=['Not Habitable', 'Habitable'])
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=balanced_ps_df, x='st_teff', y='pl_orbper', hue=Y_balanced, palette='coolwarm', alpha=0.7)
plt.title('Exoplanet Orbital Period vs. Star Temperature')
plt.xlabel('Star Temperature (K)')
plt.ylabel('Orbital Period (days)')
plt.yscale('log')
plt.legend(title='Habitability', loc='upper right', labels=['Not Habitable', 'Habitable'])
plt.grid()
plt.show()
