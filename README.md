# 🎙️ SpeechEvalAI: Whisper vs Wav2Vec2 Comparison

## 📌 Overview

SpeechEvalAI is a benchmarking project that evaluates and compares two leading Speech-to-Text (STT) models:

* **Whisper (openai/whisper-base - local model)**
* **Wav2Vec2 (facebook/wav2vec2-base-960h)**

The models are evaluated using the **Indic TTS dataset**, which contains diverse Indian language audio samples and corresponding transcripts. The project measures transcription performance using standard evaluation metrics such as Word Error Rate (WER) and Character Error Rate (CER).

---

## 🚀 Features

* 📊 Comparative analysis of Whisper and Wav2Vec2
* 🧠 Evaluation on **Indic TTS dataset**
* 📉 Metrics: Word Accuracy, Character Accuracy, WER, CER
* 📁 Custom dataset pipeline support
* ⚙️ Modular evaluation framework

---

## 🛠️ Tech Stack

* Python
* Hugging Face Transformers (Wav2Vec2, Whisper local model)
* OpenAI Whisper (API initially intended)
* Pandas, NumPy
* Librosa / SoundFile

---

## 📂 Project Structure

```id="u2bz9k"
SpeechEvalAI/
│── dataset/
│   ├── audio_files/
│   ├── dataset.csv   # Based on Indic TTS dataset
│
│── models/
│   ├── whisper_model.py
│   ├── wav2vec2_model.py
│
│── evaluation/
│   ├── metrics.py
│   ├── evaluate_models.py
│
│── results/
│   ├── output.csv
│   ├── comparison_report.txt
│
│── README.md
│── requirements.txt
```

---

## 📊 Dataset

This project uses the **Indic TTS dataset**, which includes:

* Speech samples in Indian languages
* Corresponding ground truth transcripts
* Realistic variations in accent and pronunciation

### Dataset Format

The dataset CSV must contain:

| audio_path        | transcript        |
| ----------------- | ----------------- |
| path/to/audio.wav | ground truth text |

---

## ⚙️ Installation

```bash id="k9x1sd"
git clone https://github.com/yourusername/SpeechEvalAI.git
cd SpeechEvalAI
pip install -r requirements.txt
```

---

## 🔑 Setup

### Whisper Model Note

* Initially designed to use **Whisper API**
* Due to API quota limitations, evaluation was performed using:

  * **Local model: openai/whisper-base**

No API key is required for the current setup.

---

## ▶️ Usage

Run the evaluation:

```bash id="q1c8la"
python evaluate_models.py
```

---

## 📈 Evaluation Metrics

* Word Accuracy
* Character Accuracy
* Word Error Rate (WER)
* Character Error Rate (CER)

---

## 📊 Final Results (Indic TTS Dataset)

### 🔹 Wav2Vec2 (facebook/wav2vec2-base-960h)

* **Word Accuracy:** 96.65%
* **Character Accuracy:** 98.48%
* **CER:** 1.52%

---

### 🔹 Whisper (openai/whisper-base - Local Model)

* **Word Accuracy:** 98.48%
* **Character Accuracy:** 99.37%

---

## 🔍 Key Insights

* Whisper significantly outperforms Wav2Vec2 on the **Indic TTS dataset**
* Whisper handles diverse Indian accents and speech patterns more effectively
* Wav2Vec2 performs well but is comparatively more sensitive to dataset variability
* Local Whisper model provides high accuracy without API dependency

---

## 📌 Conclusion

The experiment demonstrates that **Whisper (base model)** is more effective than **Wav2Vec2** for speech recognition tasks on the **Indic TTS dataset**, particularly in terms of transcription accuracy and robustness across diverse speech inputs.

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👤 Author

Arya A R
