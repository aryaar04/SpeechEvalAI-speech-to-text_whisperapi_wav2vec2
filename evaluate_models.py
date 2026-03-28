import pandas as pd
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
import openai
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================
OPENAI_API_KEY = "your api id"  # Required for Whisper API

# HuggingFace Model IDs
WAV2VEC2_MODEL_ID = "facebook/wav2vec2-base-960h" 
LOCAL_WHISPER_MODEL_ID = "openai/whisper-base"

# Folders to read data from
DATASET_FOLDERS = ["malyalam_female_english", "malyalam_male_english"]

# ==========================================

def load_festvox_dataset(folders):
    data = []
    for folder in folders:
        txt_file = os.path.join(folder, "txt.done.data")
        wav_dir = os.path.join(folder, "wav")
        
        if not os.path.exists(txt_file):
            print(f"Warning: {txt_file} not found.")
            continue
            
        with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("(") and line.endswith(")"):
                    # line format: ( audio_id "Transcript" )
                    inner = line[1:-1].strip()
                    first_quote_idx = inner.find('"')
                    last_quote_idx = inner.rfind('"')
                    if first_quote_idx != -1 and last_quote_idx != -1 and first_quote_idx != last_quote_idx:
                        audio_id = inner[:first_quote_idx].strip()
                        transcript = inner[first_quote_idx+1:last_quote_idx]
                        wav_path = os.path.join(wav_dir, f"{audio_id}.wav")
                        
                        if os.path.exists(wav_path):
                            data.append({"audio_path": wav_path, "transcript": transcript})
    return pd.DataFrame(data)

def load_wav2vec2():
    print(f"\nLoading Wav2Vec2 model: {WAV2VEC2_MODEL_ID}")
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_ID)
    return processor, model

def transcribe_wav2vec2(audio_path, processor, model):
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
        input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0].lower()
    except Exception as e:
        print(f"Wav2Vec2 Error on {audio_path}: {e}")
        return ""

def load_local_whisper():
    print(f"\nLoading Local Whisper model: {LOCAL_WHISPER_MODEL_ID}")
    processor = WhisperProcessor.from_pretrained(LOCAL_WHISPER_MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(LOCAL_WHISPER_MODEL_ID)
    model.config.forced_decoder_ids = None
    return processor, model

def transcribe_local_whisper(audio_path, processor, model):
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
        input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features 
        predicted_ids = model.generate(input_features)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower().strip()
    except Exception as e:
        print(f"Local Whisper Error on {audio_path}: {e}")
        return ""

def transcribe_whisper_api(audio_path, client):
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                language="en"
            )
        return transcript.text.lower().strip()
    except Exception as e:
        print(f"Whisper API Error on {audio_path}: {e}")
        return ""

def calculate_metrics(ground_truths, predictions):
    filtered_gt = []
    filtered_pred = []
    
    for gt, pred in zip(ground_truths, predictions):
        if not str(pred).strip():
            pred = " "
        filtered_gt.append(str(gt).lower().replace(".", "").replace(",", "").strip())
        filtered_pred.append(str(pred).lower().replace(".", "").replace(",", "").strip())

    word_error_rate = wer(filtered_gt, filtered_pred)
    char_error_rate = cer(filtered_gt, filtered_pred)
    
    word_accuracy = max(0.0, 1.0 - word_error_rate) * 100
    char_accuracy = max(0.0, 1.0 - char_error_rate) * 100
    
    return word_accuracy, char_accuracy

def main():
    print("Parsing dataset folders...")
    df = load_festvox_dataset(DATASET_FOLDERS)
    
    if df.empty:
        print("No valid audio files and transcripts found. Please ensure the 'wav' and 'txt.done.data' files are present in the dataset folders.")
        return

    print(f"Found {len(df)} total audio samples.")
    
    # We can trim the dataframe for a quick test if there are too many
    print("Limiting evaluation to 20 samples so it finishes quickly! Change this in the script if you want a full evaluation.")
    # df = df.head(20)     
    ground_truths = df['transcript'].tolist()

    # --- 1. Evaluate Wav2Vec2 ---
    processor_w2v2, model_w2v2 = load_wav2vec2()
    w2v2_preds = []
    print("\nTranscribing with Wav2Vec2...")
    for idx, row in df.iterrows():
        pred = transcribe_wav2vec2(row['audio_path'], processor_w2v2, model_w2v2)
        w2v2_preds.append(pred)
        print(f"[{idx+1}/{len(df)}] W2V2 processing... (Prediction: {pred[:30]}...)")


    w2v2_word_acc, w2v2_char_acc = calculate_metrics(ground_truths, w2v2_preds)
    
    # --- 2. Evaluate Local Whisper ---
    processor_wh, model_wh = load_local_whisper()
    wh_preds = []
    print("\nTranscribing with Local Whisper...")
    for idx, row in df.iterrows():
        pred = transcribe_local_whisper(row['audio_path'], processor_wh, model_wh)
        wh_preds.append(pred)
        print(f"[{idx+1}/{len(df)}] Local Whisp processing... (Prediction: {pred[:30]}...)")

    wh_word_acc, wh_char_acc = calculate_metrics(ground_truths, wh_preds)

    # --- 3. Evaluate Whisper API ---
    whisper_api_word_acc, whisper_api_char_acc = 0.0, 0.0
    if OPENAI_API_KEY == "your_openai_api_key_here":
        print("\nSkipping Whisper API evaluation because OPENAI_API_KEY is not set.")
    else:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        whisper_api_preds = []
        print("\nTranscribing with Whisper API...")
        for idx, row in df.iterrows():
            pred = transcribe_whisper_api(row['audio_path'], openai_client)
            whisper_api_preds.append(pred)
            print(f"[{idx+1}/{len(df)}] Whisper API processing... (Prediction: {pred[:30]}...)")
            time.sleep(1) # Slight delay for rate limiting

        whisper_api_word_acc, whisper_api_char_acc = calculate_metrics(ground_truths, whisper_api_preds)

    # --- Generate Final Report ---
    print("\n" + "="*50)
    print("                EVALUATION REPORT")
    print("="*50)
    
    print("\n[Wav2Vec2 Results]")
    print(f"Word-level accuracy: {w2v2_word_acc:.2f}%")
    print(f"Character-level accuracy: {w2v2_char_acc:.2f}% (derived from CER = {100-w2v2_char_acc:.2f}%)")

    print("\n[Local Whisper Results]")
    print(f"Word-level accuracy: {wh_word_acc:.2f}%")
    print(f"Character-level accuracy: {wh_char_acc:.2f}% (derived from CER = {100-wh_char_acc:.2f}%)")

    if OPENAI_API_KEY != "your_openai_api_key_here":
        print("\n[Whisper API Results]")
        print(f"Word-level accuracy: {whisper_api_word_acc:.2f}%")
        print(f"Character-level accuracy: {whisper_api_char_acc:.2f}% (derived from CER = {100-whisper_api_char_acc:.2f}%)")
        
    print("="*50)

if __name__ == "__main__":
    main()
