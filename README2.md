**MACHINE LEARNING EVALUATION OF SONG SIMILARITY USING**

**SONG SEGMENTATION CLASSIFIERS**

_Ariana Pereira_

**ABSTRACT**

This paper explores the feasibility of using

machine learning techniques to recognize

which segment of a song is most similar to

another input audio sample. By leveraging

Support Vector Machines (SVMs), and

features such as tempo, Mel-frequency

cepstral coefficients (MFCCs), chroma, and

spectral contrast, the research investigates

whether these machine learning models can

accurately identify musicological concepts

in audio clips by comparing them against

labeled training dataset. The study presents

an experimental framework for feature

extraction, model training, and evaluation,

using audio data annotated with segment

labels as an example proof of concept.

Results demonstrate the model's potential to

capture structural similarities and the basis

for evaluating song similarity to inform

future applications in music information

retrieval.

While exploring the capabilities of

generative AI music models to produce a

song based on an audio input, I questioned

to what extent the musical output was

similar to the original input, and if that input

was a copyright song, to what extent could

the AI generated output be considered

plagiarism?

To answer these questions, I looked to create

a proof-of-concept machine-learning driven

classifier to evaluate song similarity, using

an original song (with the consent of the

original composer) as the training dataset.

This paper evaluates how manual analysis of

song similarity would compare to a

machine-learning model approach. To do

this, portions of the original song were

categorized into five distinct song segments

(or labels): Verse A, Verse B, Chorus A,

Chorus B, and Outro. This paper explores

whether training a machine learning model

on certain musical features can reliably

classify song sections, with the goal of

eventually using the model to analyze how

similar AI-generated outputs adhere to the

musicological principles and patterns of an

original song used as training data.

**1\. INTRODUCTION**

Protection of musical intellectual property

from copyright infringement is a crucial part

of the music industry today, and is problem

set ripe for machine learning applications.

Especially as generative AI models can

generate brand new songs based off of

existing songs, understanding which part of

an original song the newly generated song is

replicating and to what extent the song has

been replicated is crucial to build a case for

copyright infringement if the AI model

output is too similar to the training data (i.e.

the original song).

**2\. RELATED WORK**

Significant research has been dedicated to

the segmentation and classification of

musical structures using machine learning

techniques. One notable study is

"Unsupervised Learning of Deep Features

for Music Segmentation,” by McCallum

which explores the use of unsupervised

learning to identify boundaries betweendistinct music segments, such as verses and

choruses \[2\]. The researchers highlight the

importance of feature representation in

accurately segmenting music, emphasizing

that the choice of audio features

significantly impacts the performance of

segmentation algorithms.

To this end, ensuring the right features are

used to capture identifiable characteristics of

the song section is crucial to ensure accurate

segmentation. A similar study was

performed by Lee et al. and demonstrated

the effectiveness of feature extraction

techniques utilizing long-term modulation

spectral analysis of both spectral and

cepstral features \[1\]. The spectral features

include Octave-based Spectral Contrast

(OSC), and the MPEG-7 Normalized Audio

Spectrum Envelope (NASE), while the

cepstral features involve Mel-Frequency

Cepstral Coefficients (MFCCs). The authors

introduce the concept of modulation

spectrograms, which capture the time-

varying or rhythmic information of music

signals. By decomposing each modulation

spectrum into logarithmically spaced

subbands, they compute Modulation

Spectral Contrast (MSC) and Modulation

Spectral Valley (MSV) for each subband.

Statistical aggregations of these MSCs and

MSVs across all subbands yield effective

and compact features for genre

classification, which could also be applied to

song segmentation when operating within

the context of one song. This research

highlights the significance of modulation

spectral analysis in capturing the dynamic

aspects of music signals, offering valuable

insights for tasks such as music genre

classification. The integration of spectral

and cepstral features, along with the use of

information fusion techniques, contributes to

the development of more accurate and

robust music classification systems.

Machine learning methods such as support

vector machines (SVMs) have been known

to work well for music-related classification

tasks such as genre-classification,

particularly when tempo is part of the

features extracted from the song \[3\]. The

application of SVM’s to structural

segmentation of songs remains

underexplored. McCallum employs

convolutional neural networks (CNN) to

learn deep feature embeddings for music

segmentation in an unsupervised manner \[2\]

and demonstrates that employing these

CNN-based embeddings in a classic music

segmentation algorithm significantly

enhances performance, achieving state-of-

the-art results in unsupervised music

segmentation. In the context of one song

where training data is limited, an SVM

approach is more suitable. This paper builds

on these foundations by applying machine

learning techniques, specifically SVM’s to

song segmentation within the context of one

song with the goal of exploring further

analysis of song similarity scoring.

**3\. METHODOLOGY**

**3.1 Dataset**

The dataset consisted of nine audio samples

from an original song, categorized into 5

song sections, “Verse A”, “Verse B”,

“Chorus A”, “Chorus B”, and “Outro”. Each

sample was approximately 30 seconds long

and defined by a distinct chord progression

and melody.

**3.2 Feature Extraction**

Four key audio features were extracted for

each training audio sample:• **Tempo:** Beats per minute (BPM) to

capture rhythmic characteristics.

• **MFCCs:** Representing the timbral

aspects of the audio.

• **Chroma:** Capturing harmonic

content.

• **Spectral Contrast:** Differentiating

tonal and non-tonal content.

_**See Appendix 8.1 – Feature Extraction for**_

_**further details**_

**3.3 Machine Learning Models** **Selection**

Several machine learning models were

tested to classify the audio samples by

splitting the training dataset into training and

validation (80% training / 20% validation):

1\. **K-Nearest Neighbors (K-NN):** K-

NN yielded a 100% training error on

the validation dataset.

2\. **Linear Regression:** Encoding the

five song section classifiers and

running a linear regression yielded

no improvement in accuracy vs. K-

NN (100% training error)

3\. **Support Vector Machine (SVM):**

Achieved 0% training error on the

validation dataset and was utilized

for further analysis.

4\. **Random Forest:** Given the output

of Random Forest will change each

time a new selection of features is

suppressed during training, the

model was assessed for accuracy 10

times and achieved an overall 30%

training error on the validation

dataset.

_**See Appendix 8.2 Machine Learning**_

_**Model Selection for further details**_

**3.4 Label Back testing**

Each song section category was defined by a

distinct chord progression and melody line.

When analyzed manually, the chord and

melody pairing would be key distinctive

characteristics used to discern similarity

from one song to another. To evaluate the

influence of a chord progression and melody

pairing on the model’s prediction, I re-

created each training audio sample of the

original song by playing the chord

progression with a MIDI grand piano and

played the melody line with a Vocoder

sound based on the original vocal audio.

These samples were then run through the

model to evaluate its accuracy and

understand the extent to which harmonic

content like chords and melody contributed

to the predictions.

**3.5 Feature Visualization**

Each time the model is run on an inputted

audio sample, a plotted visualization of the

features extracted from that sample is

produced to help contextualize the

prediction. Plotted visualizations for the

training dataset is available in the appendix

of this paper as well for comparison

purposes.

**4\. RESULTS**

**4.1 Initial Model Performance on the**

**Validation Dataset**

While the SVM model generated 100%

accuracy on the validation dataset, the

visualizations of the nine training data points

helped to illuminate artifacts and outliers

from the feature extraction process. These

factors likely contributed to skewed results

when testing the model on other audio

samples outside of the training dataset.Most notably, the average tempo extracted

across the training dataset was measured at

~125 bpm which was not intended. The song

was composed at ~80 bpm, but the original

audio recording was recorded without a

metronome and features a syncopated chord

striking pattern, likely influencing the

irregular tempo reading.

**Training Audio**

**Sample**

**Tempo in**

**BPM**

Verse 1 A 119.12

Verse 2 A 120.47

Verse 1 B 128.73

Verse 2 B 131.85

Chorus 1 A 131.40

Chorus 2 A 133.49

Chorus 1 B 122.19

Chorus 2 B 126.6

Outro 109.46

**Average Tempo 124.81**

**Fig. 1.** Extracted tempo in BPM per audio

sample in the training dataset.

_**See Appendix 8.3 Training Audio Sample**_

_**Extracted Feature Visualization for further**_

_**details**_

**4.2 Chord and Melody-Only Analysis**

Testing the SVM model on MIDI samples of

isolated chord progression and melody line

revealed high testing errors. Most samples

were misclassified as “Verse B”, likely

because the chord striking pattern in Verse

B is much less frequent and therefore more

aligns with the way the MIDI sample was

played, each chord struck only once.

The remaining error likely resulted from

noise in the feature extraction data. Some

notable issues which have arisen:

• **Tempo.** In the MIDI recordings, the

tempo was consistently set to 80 bpm

and the notes were quantized.

However, the tempo extracted from

the audio clips averaged closer to

~122 bpm.

• **Chroma**. The chroma features in the

MIDI dataset also varied greatly

compared to their respective

counterparts in the training dataset,

likely due to differences in the chord

voicings and inversions used. For

example, the Chroma visualization

for Verse A in the MIDI sample

indicated a strong presence of the C

pitch, whereas the training dataset

shows a heavier emphasis on the E

pitch throughout Verse A. A similar

phenomenon occurred when

comparing Chorus A across the

training and testing data sets.

• **MFCC.** While MFCC’s have been

known for their usefulness in speech

recognition, the Vocoder melody

lacks lyrical diction which is found

in the original audio recording and

training dataset and is likely

contributing to the incorrect

classifications.

This indicates that while chord progression

and melody are important features for a

human observer, it alone does not

sufficiently distinguish between song

sections using the selected features.

**MIDI Sample**

**Section**

**Extracted**

**Tempo Prediction**

Verse B

Verse B

Verse B

Verse A 98.80 Verse B 130.55 Chorus A 114.84 Chorus B 127.89 Outro 137.29 Verse B

Verse B

**Total Dataset**

**Average**

**Tempo:**

**121.87**

**Testing**

**Error:**

**80%Fig. 2.** Extracted tempo in BPM and

classifier prediction from the SVM model

_**See to Appendix 8.4 MIDI Sample**_

_**Extracted Feature Visualization for further**_

_**details**_

**5\. Discussion**

The results demonstrate that machine

learning can identify recurring patterns in

musical structures, though challenges

remain in refining the features extracted to

achieve more granular accuracy. The high

error rate in chord and melody-only analysis

suggests that chord voicing and rhythmic

nuances play critical roles in distinguishing

song sections when using the current set of

features. Moreover, anomalies like the

higher-than-expected BPM underscore the

need for refining feature extraction

techniques.

Future work could explore:

• Expanding the dataset to include

more granular song subsections

• Recreating this analysis on another

piece of music

• Incorporating deep learning models

such as convolutional neural

networks (CNNs) for more advanced

feature learning

• Refining and expanding feature

extraction to account for dynamics

and phrasing in music or tailoring

features to the style of music being

tested

• Incorporating a linear regression

model on top of the existing

classifiers to expose the probability

of each classifier and demonstrate

how similar an inputted audio

sample is to the overall original

training dataset

• Incorporating semantic data from

chord charts and lyrics as part of

feature extraction _(See Appendix 8.5_

_– Original Composition Chord Chart_

_and Lyrics for reference)_

**6\. Conclusion**

This study highlights the potential of

machine learning in music segmentation

tasks, providing insights into the structural

patterns of songs. While the SVM model

showed promise, the findings underscore the

importance of integrating multiple musical

features for robust classification. These

insights pave the way for developing tools

that assist musicians and composers in

analyzing and enhancing their work.

**7\. References**

\[1\] Chang-Hsing Lee, Janez Kecic, and

Jeng-Shyang Pan. “Automatic Music

Genre Classification Based on

Modulation Spectral Analysis of

Spectral and Cepstral Features.” _IEEE_

_Transactions on Multimedia_, vol. 11,

no. 4, 2009, pp. 717-729.

https://doi.org/10.1109/TMM.2009.20

22917.

\[2\] Andrew McCallum. “Unsupervised

Learning of Deep Features for Music

Segmentation.” _International Society_

_for Music Information Retrieval_, 2019,

pp. 467-472.

\[3\] R. Thiruvengatanadhan. “Music

Genre Classification Using SVM -

IRJET.” _International ResearchJournal of Engineering and_

_Technology (IRJET).,_ vol. 5 pp. 1058-

1060, October 2018.

**8\. Appendix**

**8.1 Feature Extraction**

**8.2 Machine Learning Model Selection**

_K-NN Method_

_K-NN ResultLinear Regression Approach_

_Linear Regression Result_

_SVM Approach_

_SVM ResultRandom Forest Approach_

_Random Forest Result – over the course of 10 code executions_**8.3 Training Audio Samples Extracted Feature Visualization**

_Verse 1 A_

_Verse 2 AVerse 1 B_

_Verse 2 BChorus 1 A_

_Chorus 2 AChorus 1 BChorus 2 B_

_Outro_

**8.4 MIDI Sample Extracted Feature Visualization**_Verse A_

_Verse BChorus AChorus B_

_Outro_

**8.5 Original Composition Chord Chart and Lyrics**Lyrics by Kylie Lefkowitz

Musical Composition by Avina Pereira

_**Verse 1 A**_

_Csus C Csus C Am G_

_\[Instrumental\]_

_Csus C_

_Once upon a time_

_Csus C_

_I became too sentimental_

_Am_

_I got lost inside my mind_

_G_

_Too nostalgic for what was_

_**Verse 1 B**_

_Csus C_

_I couldn’t see the light_

_Csus C_

_In the bitter New York winters_

_Am_

_Withdrawn into myself_

_G_

_I was alone_

_F Fm_

_Forced to surrender_

_**Chorus 1 A**_

_C/E_

_I’ve loved you all my life_

_Am F Fm_

_All our past lives together and this one_

_C G_

_Once upon a time and the time after time before_

_**Chorus 1 B**_

_F Fm C G_

_I gave you everything, orbiting, and I know that I’ll give it again_

_F Fm_

_we’ll collide_

_C_

_When we’re young again_

_**Verse 2 A**_

_Csus COnce upon a time_

_Csus C_

_I became too sentimental_

_Am_

_In the carnage of my heart_

_G_

_Estranged from what it was_

_**Verse 2 B**_

_Csus C_

_You’re a beacon in the night_

_Csus C_

_And a warmth in all the wreckage_

_Am G_

_The tides are rolling in and I can see_

_F Fm_

_The ties won’t sever it feels_

_Like now or never_

_**Chorus 2 A**_

_C/E_

_I’ve loved you all my life_

_Am F Fm_

_All our past lives together and this one_

_C G_

_I’ve known you a thousands years And I’ll know you 1000 more_

_**Chorus 2 B**_

_F Fm C G_

_Just take my hand and spin, orbiting, Spiral round as we start it again_

_F Fm_

_we’ll collide_

_C Am_

_When we’re young again_

_**Outro**_

_F Fm_

_It makes sense_

_C_

_That it’s you my friend_

_F Fm C_

_We’ll collide when we’re young again_
