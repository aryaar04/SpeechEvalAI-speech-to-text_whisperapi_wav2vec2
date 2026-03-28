# 🎙️ SpeechEvalAI: Whisper vs Wav2Vec2 Comparison

## 📌 Overview

SpeechEvalAI is a benchmarking project that evaluates and compares two leading Speech-to-Text (STT) models:

* **Whisper (openai/whisper-base - local model)**
* **Wav2Vec2 (facebook/wav2vec2-base-960h)**

The project measures transcription performance on an Indic TTS dataset using standard evaluation metrics such as Word Error Rate (WER) and Character Error Rate (CER).

---

## 🚀 Features

* 📊 Comparative analysis of Whisper and Wav2Vec2
* 🧠 Fine-tuned evaluation on Indic TTS dataset
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

```
SpeechEvalAI/
│── dataset/
│   ├── audio_files/
│   ├── dataset.csv
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

## 📊 Dataset Format

The dataset CSV must contain:

| audio_path        | transcript        |
| ----------------- | ----------------- |
| path/to/audio.wav | ground truth text |

---

## ⚙️ Installation

```bash
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

```bash
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

* Whisper significantly outperforms Wav2Vec2 in both word-level and character-level accuracy
* Whisper demonstrates better robustness for Indic speech and pronunciation variations
* Wav2Vec2 performs well but is comparatively more sensitive to dataset characteristics
* Local Whisper model provides high accuracy without API dependency

---

## 📌 Conclusion

The experiment demonstrates that **Whisper (base model)** is more effective than **Wav2Vec2** for speech recognition tasks on Indic datasets, particularly in terms of transcription accuracy and reliability.

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👤 Author

Arya A R
