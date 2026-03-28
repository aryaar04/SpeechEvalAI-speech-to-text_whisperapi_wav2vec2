# Speech to Text Word-Level & Character-Level Accuracy Evaluator

This project evaluates the performance of Wav2Vec2 and Whisper models on an Indic TTS English dataset. It provides **Word-Level Accuracy** and **Character-Level Accuracy** metrics, similar to the CNN LSTM evaluation format you used before.

## Prerequisites

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your dataset**:
   You must create a file named `dataset.csv` in this folder with your dataset metadata. It should have the following two columns:
   - `audio_path`: The relative or absolute path to the audio file (e.g., `.wav`, `.mp3`).
   - `transcript`: The ground truth English text (what was actually spoken).

   *Example `dataset.csv`:*
   ```csv
   audio_path,transcript
   audio/sample1.wav,this is a test sentence
   audio/sample2.wav,the weather is nice today
   ```

3. **(Optional) OpenAI API Key**:
   If you want to evaluate using the paid **Whisper API** (which is usually much faster and more accurate than running locally), open `evaluate_models.py` and replace:
   ```python
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```
   with your actual OpenAI API key. If you leave it as-is, the script will skip the Whisper API step but still run the **local Open-Source Whisper** and **Wav2Vec2** models.

## How to Run

Execute the script from your terminal:
```bash
python evaluate_models.py
```

## How It Works

It uses the `jiwer` package to calculate Word Error Rate (WER) and Character Error Rate (CER).
- **Word-level accuracy** = `100 - (WER * 100)`
- **Character-level accuracy** = `100 - (CER * 100)`

By default, the script compares:
1. **Wav2Vec2 Base 960h** (`facebook/wav2vec2-base-960h`)
2. **Whisper Base (Local/Open Source)** (`openai/whisper-base`)
3. **Whisper API (Cloud)** (`whisper-1` via OpenAI)

If you have a fine-tuned Wav2Vec2 model for Indian English, you can change the `WAV2VEC2_MODEL_ID` at the top of the script!
