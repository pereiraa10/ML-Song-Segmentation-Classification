# **Machine Learning Evaluation Of Song Similarity Using Song Segmentation Classifiers**

## Abstract
-------
This repository explores the feasibility of using machine learning techniques to recognize which segment of a song is most similar to another input audiosample. By leveraging Support Vector Machines (SVMs), and features such as tempo, Mel-frequency cepstral coefficients (MFCCs), chroma, and spectral contrast, the research investigates whether these machine learning models can accurately identify musicological concepts in audio clips by comparing themagainst labeled training dataset. The study presents an experimental frameworkfor feature extraction, model training, and evaluation, using audio dataannotated with segment labels as an example proof of concept. Resultsdemonstrate the model's potential to capture structural similarities and thebasis for evaluating song similarity to inform future applications in musicinformation retrieval.

# Get Started

## Get the project from GitHub:

    git clone https://github.com/pereiraa10/ML-Song-Segmentation-Classification/

## Enter the project

```bash
    cd ML-Song-Segmentation-Classification/
```

## Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run each of the algorithm's respective experiments as referenced in section **3.3 Machine Learning Models Selection**.
For example:

```bash
python knn.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
