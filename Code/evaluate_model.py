import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('your_model.h5')

# Define a dictionary for class labels if used during training
class_labels = {0: 'anger', 1: 'boredom', 2: 'disgust', 3: 'anxiety', 4: 'happiness', 5: 'sadness', 6: 'neutral'}

# Preprocessing function (adapt to match training preprocessing)
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean.reshape(1, -1)

# Prediction function
def predict_emotion(file_path):
    processed_audio = preprocess_audio(file_path)
    prediction = model.predict(processed_audio)
    predicted_class = np.argmax(prediction, axis=1)[0]
    print("Predicted Emotion:", class_labels[predicted_class])

if __name__ == "__main__":
    # Test with a sample audio file path
    test_audio_path = '../wav/03a01Fa.wav'
    predict_emotion(test_audio_path)
