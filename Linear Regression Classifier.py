import librosa
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Function to extract features from a single audio file
def extract_features(file_path, sr=22050, n_mfcc=13):
    # Load audio file
    y, sr = librosa.load(file_path, sr=sr)
    
    # Feature extraction
    features = {}
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    features['mfcc_std'] = np.std(mfccs, axis=1)
    
    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma, axis=1)
    features['chroma_std'] = np.std(chroma, axis=1)
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['contrast_mean'] = np.mean(contrast, axis=1)
    features['contrast_std'] = np.std(contrast, axis=1)
    
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
    features['tonnetz_std'] = np.std(tonnetz, axis=1)
    
    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    
    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    
    return features

# Directory containing audio files
audio_dir = "/Users/arianapereira/Documents/Masters/Documents for Application/project/suno_songs"

# Initialize an empty list to store feature dictionaries
features_list = []

# Loop over files in the directory
for file_name in os.listdir(audio_dir):
    file_path = os.path.join(audio_dir, file_name)
    if file_path.endswith(".mp3") or file_path.endswith(".wav"):
        print(f"Processing {file_name}...")
        features = extract_features(file_path)
        features['file_name'] = file_name  # Add file name to the features
        features_list.append(features)

# Convert to a DataFrame
features_df = pd.DataFrame(features_list)

# Save features to a CSV
features_df.to_csv("audio_features.csv", index=False)
print("Features saved to audio_features.csv")


##############



# Step 1: Load the dataset
data = pd.read_csv("audio_features.csv")

# Step 2: Check for target column
if 'similarity_score' not in data.columns:
    raise ValueError("The dataset must contain a 'similarity_score' column.")

# Step 3: Split features (X) and target (y)
X = data.drop(columns=["file_name", "similarity_score"])  # Drop non-numeric columns
y = data["similarity_score"]

# Step 4: Handle missing values (if any)
X.fillna(X.mean(), inplace=True)

# Step 5: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Step 9: Predict similarity scores for new songs
new_song_features = np.random.rand(5, X.shape[1])  # Simulating new song features
new_song_scaled = scaler.transform(new_song_features)
predicted_scores = model.predict(new_song_scaled)

print("Predicted Similarity Scores for New Songs:", predicted_scores)
