# Speech-Emotion-Recognition
Classifies the emotion in a short speech clip (angry, disgust, fear, happy, neutral, sad, surprise) using audio features and two models: a classical SVM baseline and a 1D CNN. Includes an interactive Streamlit dashboard for live prediction, model comparison and error analysis.  Course project: Natural Language Processing / Speech Processing.
# Speech Emotion Recognition (SER)

Classifies the emotion in a short speech clip (angry, disgust, fear, happy, neutral, sad, surprise) using audio features and two models: a classical **SVM** baseline and a **1D CNN**. Includes an interactive **Streamlit dashboard** for live prediction, model comparison and error analysis.

> Course project: Natural Language Processing / Speech Processing.

---
<img width="1918" height="999" alt="image" src="https://github.com/user-attachments/assets/5a02f87d-0caa-49b9-859e-a1741ee950c5" />
<img width="1919" height="994" alt="image" src="https://github.com/user-attachments/assets/d0a84ee6-6eae-46b2-8966-fa5e7b4fc2f8" />
## live url:https://outlined-missile-fellowship-watt.trycloudflare.com/
## Features
- Upload a WAV file or record from the microphone and get a predicted emotion
- Side-by-side **SVM vs CNN** predictions with per-emotion probabilities
- Waveform and MFCC heatmap of the input audio
- Model comparison: accuracy, macro precision / recall / F1, CNN training curve
- Per-emotion F1 scores and row-normalised confusion matrices

## Dataset
Kaggle: [Speech Emotion Recognition (en)](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en), a combination of four public English datasets.

| Dataset | Speakers | Notes |
|---|---|---|
| CREMA-D | 91 | Most clips, diverse speakers; no "surprise" class |
| TESS | 2 | Female speakers only |
| SAVEE | 4 | Male speakers only |
| RAVDESS | 24 | "calm" class removed to match the others |

Labels are unified into 7 classes: `angry, disgust, fear, happy, neutral, sad, surprise`.
Labels are read from file names (no CSV is provided).

**Important:** all four datasets are *acted* emotions recorded in studio conditions.

## Method
1. **Preprocessing:** resample to 16 kHz, mono, first 3 seconds of each clip.
2. **Features:** 40 MFCCs.
   - SVM: mean and standard deviation over time (80 values per clip)
   - CNN: full MFCC sequence (100 time steps x 40 coefficients), standardised with training-set mean/std
3. **Split:** `GroupShuffleSplit` by speaker (80/20), so no speaker appears in both train and test. This gives an honest, speaker-independent evaluation.
4. **Models**
   - SVM: StandardScaler + RBF SVM (C=10)
   - CNN: 3 x Conv1D (64 / 128 / 256 filters) with BatchNorm, MaxPooling and Dropout, global average pooling, dense layer, softmax

## Results (speaker-independent test set, 1,911 clips)

| Model | Features | Accuracy | Macro F1 |
|---|---|---|---|
| SVM | MFCC mean + std | 48.7% | 0.48 |
| CNN | MFCC sequence | *TODO: fill in* | *TODO* |

SVM per-emotion F1: angry 0.63, sad 0.57, neutral 0.48, fear 0.44, happy 0.42, surprise 0.46, disgust 0.38.

Random guessing over 7 classes is about 14%.

Observations:
- Angry and sad are easiest (strong energy / pitch differences).
- Disgust, fear and happy are most often confused with each other.
- "surprise" has only 39 test clips (CREMA-D has none), so its score is not reliable.

## Project structure
```
.
├── app.py              # Streamlit dashboard
├── emotion_cnn.keras   # trained CNN
├── svm.joblib          # trained SVM pipeline
├── mean.npy, std.npy   # CNN input normalisation
├── metrics.json        # stored evaluation results used by the dashboard
├── requirements.txt
├── packages.txt        # libsndfile1, ffmpeg (for Streamlit Cloud)
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Use Python 3.10 or 3.11 (TensorFlow compatibility). Open http://localhost:8501.

`requirements.txt` should pin the same versions used to train the model:
```
tensorflow==<version>
scikit-learn==<version>
librosa==<version>
numpy==<version>
pandas==<version>
plotly
joblib
streamlit
```

## Deploy
Push the files above to a public GitHub repo and deploy `app.py` on [Streamlit Community Cloud](https://share.streamlit.io).
`packages.txt` must contain `libsndfile1` (and `ffmpeg` if you accept `.ogg` / `.mp3` uploads).

## Limitations
- Acted, English-only training data; performance on spontaneous speech, other languages, accents or noisy phone recordings is expected to be much lower.
- Only the first 3 seconds of audio are used.
- Class imbalance across datasets (no CREMA-D "surprise"; neutral is smaller).
- Test set contains a limited number of speakers, so the score can vary with the split.
- Emotion labels are subjective.

## Future work
- Fine-tune Wav2Vec2 / HuBERT on raw audio
- Data augmentation (noise, pitch shift, speed)
- Per-speaker normalisation
- Urdu / code-switched emotion data
- Combine audio emotion with text sentiment from Whisper transcripts

## Acknowledgements
CREMA-D, TESS, SAVEE and RAVDESS dataset authors; the Kaggle dataset by dmitrybabko; librosa, scikit-learn, TensorFlow/Keras, Streamlit and Plotly.
