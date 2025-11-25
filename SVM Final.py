import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

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

# Train SVM Model
model=SVC(kernel='linear')
model.fit(X_train, y_train)

# Save the model and scaler
with open('svm_audio_classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)


# Prediction function
def predict_audio_section_with_visualization(file_path):
    """
    Predict the section of an audio file and display visualizations (spectrogram and features).
    """

    # Load scaler and model
    with open('svm_audio_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Extract features and scale
    y, sr = librosa.load(file_path, duration=30)
    features = extract_features(file_path)
    features_scaled = scaler.transform([features])  # Scale features
    
    # Predict
    prediction = model.predict(features_scaled)


    # Compute Spectrogram and Chroma features for visualization
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S), ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Plot visualizations
    fig, ax = plt.subplots(nrows=3, figsize=(12, 12))


    # Spectrogram Chart
    img1 = librosa.display.specshow(S_db, y_axis='log', x_axis='time', sr=sr, ax=ax[0])
    fig.colorbar(img1, ax=ax[0], format='%+2.0f dB')
    ax[0].set_title('Power Spectrogram')


    # Chroma Features Chart
    img2 = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, ax=ax[1], cmap='coolwarm')
    fig.colorbar(img2, ax=ax[1])
    ax[1].set_title('Chroma Features')


    # Plot Tempo 
    avg_tempo = features[0]
    tempo_x = [0]
    ax[2].bar(tempo_x, [avg_tempo], color='g', label='Tempo')
    ax[2].text(tempo_x[0], avg_tempo + 0.5, f"{avg_tempo:.2f}", ha='center', va='bottom', fontsize=10, color='black')
    ax[2].set_title('Average Tempo')
    ax[2].legend(loc='upper right', fontsize='small')

    
    # Show the plots
    plt.tight_layout()
    plt.show()
    
    return prediction[0]


file_path = input('Please enter the path to the audio file:')

try:
    prediction = predict_audio_section_with_visualization(file_path)
    print(f"The predicted section of the audio file is: {prediction}")


except Exception as e:
    print(f"An error occurred: {e}")


