# **Machine Learning Evaluation Of Song Similarity Using Song Segmentation Classifiers**

## Overview

This repository explores the feasibility of using machine learning techniques to identify which segment of a song is most similar to another input audiosample. Leveraging Support Vector Machines (SVMs), and features such as tempo, Mel-frequency cepstral coefficients (MFCCs), chroma, and spectral contrast, the research investigates whether these machine learning models can accurately identify musicological concepts in audio clips by comparing themagainst labeled training dataset. The study presents an experimental frameworkfor feature extraction, model training, and evaluation, using audio dataannotated with segment labels as an example proof of concept. Resultsdemonstrate the model's potential to capture structural similarities and thebasis for evaluating song similarity to inform future applications in musicinformation retrieval.

-------

# Get Started

## Clone the Project:

    git clone https://github.com/pereiraa10/ML-Song-Segmentation-Classification/

## Enter the Project Directory:

```bash
cd ML-Song-Segmentation-Classification/
```

## Install Dependencies:

```bash
pip install -r requirements.txt
```

-------

## Usage

To run the experiments described in section **3.3 Machine Learning Models Selection**, execute the corresponding script.
For example:

```bash
python knn.py
```

To run the SVM prediction tool described in section **4.2 Chord and Melody-Only Analysis**, execute `svm_midi_analysis.py`. When prompted, provide the relative file path to the audio file in the `MIDI_testset` folder that you want to analyze. 

```bash
python svm_midi_analysis.py
```

-------

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

