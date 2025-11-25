import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Load audio file and extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path,duration=30)
    n_fft= min(2048,len(y))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    avg_tempo = np.mean(dtempo)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft).T, axis=0)
    return np.hstack([avg_tempo, mfcc, chroma, spectral_contrast])

# Example dataset (file paths and labels)

files = ['/Users/arianapereira/Documents/Masters/Documents for Application/project/Collide clips/Collide Chorus 1a.mp3',
         '/Users/arianapereira/Documents/Masters/Documents for Application/project/Collide clips/Collide Chorus 1b.mp3',
         '/Users/arianapereira/Documents/Masters/Documents for Application/project/Collide clips/Collide Chorus 2a.mp3',
         '/Users/arianapereira/Documents/Masters/Documents for Application/project/Collide clips/Collide Chorus 2b.mp3',
         '/Users/arianapereira/Documents/Masters/Documents for Application/project/Collide clips/Collide Outro.mp3',
         '/Users/arianapereira/Documents/Masters/Documents for Application/project/Collide clips/Collide Verse 1a.mp3',
         '/Users/arianapereira/Documents/Masters/Documents for Application/project/Collide clips/Collide Verse 1b.mp3',
         '/Users/arianapereira/Documents/Masters/Documents for Application/project/Collide clips/Collide Verse 2a.mp3',
         '/Users/arianapereira/Documents/Masters/Documents for Application/project/Collide clips/Collide Verse 2b.mp3'] 
labels = ['Chorus A', 'Chorus B','Chorus A', 'Chorus B', 'Outro', 'Verse A', 'Verse B','Verse A', 'Verse B']

# Extract features
features = np.array([extract_features(file) for file in files])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train SVM Model
model=SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("Predictions:", y_pred[:])
print("True values:", y_test[:])

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


# OUTPUT:
#   Predictions: ['Verse' 'Chorus2']
#   True values: ['Verse', 'Chorus2']
#   Accuracy: 1.00
