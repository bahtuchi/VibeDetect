import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, classification_report
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.metrics import AUC
import librosa
import librosa.display
from tqdm import tqdm  # For progress tracking
import warnings
warnings.filterwarnings('ignore')

# Load dataset paths and labels
paths = []
labels = []
for dirname, _, filenames in os.walk('../wav'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1].split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break

# Emotion codes and their meanings
emotion_codes = {
    'W': 'anger',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'anxiety',
    'F': 'happiness',
    'T': 'sadness',
    'N': 'neutral'
}

# Load file paths and labels into a DataFrame
df = pd.DataFrame({'speech': paths, 'label': labels})
df['label'] = df['label'].apply(lambda x: emotion_codes.get(x, 'Unknown'))

# Feature extraction function with MFCC
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Extract MFCC features
X_mfcc = np.array([extract_mfcc(path) for path in tqdm(df['speech'], desc="Extracting MFCCs")])

# Apply PCA for dimensionality reduction
pca = PCA(n_components=20)  # Reducing to 20 components for efficiency
X_pca = pca.fit_transform(X_mfcc)

# Reshape data for LSTM input
X = np.expand_dims(X_pca, -1)

# One-hot encode labels
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']]).toarray()

# Model architecture with LSTM and AUC metric
model = Sequential([
    LSTM(128, input_shape=(X.shape[1], 1), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(emotion_codes), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[AUC(name="auc")])
model.summary()

# Training the model
history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)

# Predict probabilities for test data and calculate AUC-ROC
y_pred_proba = model.predict(X)
y_true = y
auc_roc = roc_auc_score(y_true, y_pred_proba, average="macro", multi_class="ovr")

print(f"AUC-ROC Score (Macro-Averaged): {auc_roc}")

# Optional: Classification Report
y_pred = np.argmax(y_pred_proba, axis=1)
y_true_classes = np.argmax(y_true, axis=1)
report = classification_report(y_true_classes, y_pred, target_names=enc.categories_[0])
print("Classification Report:")
print(report)
