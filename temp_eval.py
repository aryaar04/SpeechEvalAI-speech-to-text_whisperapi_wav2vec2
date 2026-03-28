import pandas as pd
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
import os

DATASET_FOLDERS = ["malyalam_female_english", "malyalam_male_english"]

def load_festvox_dataset(folders):
    data = []
    for folder in folders:
        txt_file = os.path.join(folder, "txt.done.data")
        wav_dir = os.path.join(folder, "wav")
        if not os.path.exists(txt_file): continue
        with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("(") and line.endswith(")"):
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

def load_local_whisper():
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.config.forced_decoder_ids = None
    return processor, model

def transcribe_local_whisper(audio_path, processor, model):
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
        input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features 
        predicted_ids = model.generate(input_features)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower().strip()
    except Exception as e:
        return ""

def main():
    df = load_festvox_dataset(DATASET_FOLDERS)
    df = df.head(20)
    ground_truths = df['transcript'].tolist()
    
    processor_wh, model_wh = load_local_whisper()
    wh_preds = []
    for idx, row in df.iterrows():
        wh_preds.append(transcribe_local_whisper(row['audio_path'], processor_wh, model_wh))

    filtered_gt = [str(gt).lower().replace(".", "").replace(",", "").strip() for gt in ground_truths]
    filtered_pred = [str(pred).lower().replace(".", "").replace(",", "").strip() for pred in wh_preds]

    w_err = wer(filtered_gt, [p if p else " " for p in filtered_pred])
    c_err = cer(filtered_gt, [p if p else " " for p in filtered_pred])
    
    w_acc = max(0.0, 1.0 - w_err) * 100
    c_acc = max(0.0, 1.0 - c_err) * 100
    
    print(f"WHISPER_WORD_ACC={w_acc:.2f}")
    print(f"WHISPER_CHAR_ACC={c_acc:.2f}")

if __name__ == "__main__":
    main()
