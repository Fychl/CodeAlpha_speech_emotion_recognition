import os
import math
import json
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import delayed, Parallel
import pyloudnorm as pyln
import IPython.display as ipd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T
import librosa.display

warnings.filterwarnings("ignore")

# Dataset paths
CREMAD = "C:/Users/Utkarsh/OneDrive/Desktop/CodeAlpha_ML/speech_emotion_recognition/Crema/"
RAVDESS = "C:/Users/Utkarsh/OneDrive/Desktop/CodeAlpha_ML/speech_emotion_recognition/archive (1)/Actor_01"
SAVEE = "C:/Users/Utkarsh/OneDrive/Desktop/CodeAlpha_ML/speech_emotion_recognition/SAVEE/ALL"
TESS = "C:/Users/Utkarsh/OneDrive/Desktop/CodeAlpha_ML/speech_emotion_recognition/TESS Toronto emotional speech set data/"
MLEND = "C:/Users/Utkarsh/OneDrive/Desktop/CodeAlpha_ML/speech_emotion_recognition/mlend-spoken-numerals/MLEndSND_Public/MLEndSND_Public/"
IEMOCAP = "C:/Users/Utkarsh/OneDrive/Desktop/CodeAlpha_ML/speech_emotion_recognition/iemocapfullrelease/IEMOCAP_full_release/"

OUTPUT_DIR = "processed_audio"

import pandas as pd
audios = pd.DataFrame()

# Helper functions
def process_cremad_file(file):
    emotion_map = {"DIS": "Disgust", "SAD": "Sad", "HAP": "Happy",
                   "NEU": "Neutral", "FEA": "Fearful", "ANG": "Angry"}
    emotion = emotion_map[file.split("_")[2]]
    return {"Dataset": "CREMA-D", "Path": CREMAD + file, "File": "".join(file.split(".")[:-1]), "Emotion": emotion}

def process_ravdess_files():
    emotion_map = {1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad", 5: "Angry", 6: "Fearful", 7: "Disgust", 8: "Surprised"}
    rows = []
    for folder in os.listdir(RAVDESS):
        folder_path = os.path.join(RAVDESS, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            emotion = emotion_map[int(file.split("-")[2])]
            rows.append({"Dataset": "RAVDESS", "Path": os.path.join(folder_path, file), "File": "".join(file.split(".")[:-1]), "Emotion": emotion})
    return rows

def process_tess_files():
    emotion_map = {"fear": "Fearful", "happy": "Happy", "disgust": "Disgust", "sad": "Sad",
                   "ps": "Surprised", "angry": "Angry", 'neutral': "Neutral"}
    rows = []
    for folder in os.listdir(TESS):
        folder_path = os.path.join(TESS, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            emotion = emotion_map[file.split("_")[2].split(".")[0]]
            rows.append({"Dataset": "TESS", "Path": os.path.join(folder_path, file), "File": "".join(file.split(".")[:-1]), "Emotion": emotion})
    return rows

def process_savee_files():
    emotion_map = {'a': "Angry", 'd': "Disgust", 'f': "Fearful", 'h': "Happy",
                   'n': "Neutral", 'sa': "Sad", 'su': "Surprised"}
    rows = []
    for file in os.listdir(SAVEE):
        # Extract emotion code from filename, e.g. 'a', 'd', 'f', etc.
        # Filename format example: 'DC_a01.wav' or similar
        # Split by underscore and take the second part, then remove digits and extension
        emotion_code = ''.join(filter(str.isalpha, file.split("_")[1]))
        emotion = emotion_map.get(emotion_code, "Unknown")
        rows.append({"Dataset": "SAVEE", "Path": os.path.join(SAVEE, file), "File": "".join(file.split(".")[:-1]), "Emotion": emotion})
    return rows

def process_iemocap_session(iemocap_path, sess_no):
    LABELS_PATH = os.path.join(iemocap_path, f"Session{sess_no}/dialog/EmoEvaluation/Categorical/")
    AUDIOS_PATH = os.path.join(iemocap_path, f"Session{sess_no}/sentences/wav/")
    accepted_emotions = ["Happiness", "Anger", "Calm", "Neutral", "Sadness", "Disgust", "Surprise",
                         "Fear", "Frustration", "Excited"]
    emotion_map = {'anger': "Angry", 'disgust': "Disgust", 'fear': "Fearful", 'happiness': "Happy",
                   "frustration": "Frustration", 'neutral state': "Neutral", 'sadness': "Sad", 'surprise': "Surprised",
                   'excited': "Excited"}
    res_df = pd.DataFrame(columns=['Dataset', "Path", "Emotion", "File"])
    for file in tqdm(os.listdir(LABELS_PATH), total=len(os.listdir(LABELS_PATH)), desc=f'Session {sess_no}'):
        if file.endswith("_cat.txt"):
            with open(os.path.join(LABELS_PATH, file), "r") as f:
                for line in f.readlines():
                    wavfile, *emotions = line.split(" :")
                    session_id = "_".join(wavfile.split("_")[:-1])
                    path = os.path.join(AUDIOS_PATH, session_id, wavfile + ".wav")
                    if (wavfile + ".wav") not in os.listdir(os.path.join(AUDIOS_PATH, session_id)):
                        print(f"Error, {session_id}")
                        continue
                    try:
                        emotion = [emotion for emotion in emotions for accepted_emotion in accepted_emotions if accepted_emotion in emotion][0]
                    except IndexError:
                        break
                    emotion = emotion.replace("\n", "").split(";")[0]
                    if path not in audios.Path.values:
                        res_df.loc[len(res_df)] = {"Dataset": "IEMOCAP", "Path": path, "File": wavfile, "Emotion": emotion_map[emotion.lower()]}
    return res_df

def estimate_snr(y):
    signal_power = np.mean(y ** 2)
    noise_power = np.var(y - np.mean(y))
    return 10 * np.log10(signal_power / (noise_power + 1e-12))

def process_file(idx, row, meter):
    audio_path = row['Path']
    try:
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        loudness = meter.integrated_loudness(y)
        rms = np.sqrt(np.mean(y ** 2))
        peak = np.max(np.abs(y))
        signal_to_noise = estimate_snr(y)
        silence_pct = np.sum(np.abs(y) < 0.001) / len(y)
        return {
            "ID": idx,
            'path': audio_path,
            "dataset": row['Dataset'],
            'duration': duration,
            'sample_rate': sr,
            'loudness': loudness,
            'rms': rms,
            'peak': peak,
            "signal_to_noise": signal_to_noise,
            "silence_pct": silence_pct,
            'emotion': row['Emotion']
        }
    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {e}")
        return idx

def get_info(df, sample_size=5, meter=None):
    if meter is None:
        meter = pyln.Meter(22050)
    print("🔍 Verifying audio files...")
    results = Parallel(n_jobs=-1)(
        delayed(process_file)(idx, row, meter) for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking audio files")
    )
    audio_info = [r for r in results if isinstance(r, dict)]
    invalid_files = [r for r in results if not isinstance(r, dict)]
    print(f"✅ Valid files: {len(audio_info)}")
    print(f"❌ Invalid files: {len(invalid_files)}")
    if len(audio_info) == 0:
        print("⚠️  No valid audio files found.")
        return None, None
    sr_df = pd.DataFrame(audio_info)
    return sr_df, invalid_files

TARGET_SR = 22050
TARGET_RMS = 0.03
MAX_PEAK = 0.99
DURATION = 5.0
TARGET_dBFS = -23.0
NUM_SAMPLES = int(TARGET_SR * DURATION)
MIN_SAMPLES = TARGET_SR * 2  # Min samples of 2 seconds

def normalize_audio(audio_data, target_dBFS=TARGET_dBFS):
    rms = torch.sqrt(torch.mean(audio_data ** 2))
    if rms == 0:
        return audio_data
    current_dBFS = 20 * torch.log10(rms)
    gain_dB = target_dBFS - current_dBFS
    gain_linear = 10 ** (gain_dB / 20)
    normalized_audio = audio_data * gain_linear
    return normalized_audio

def process_row(idx, row):
    path = row['Path']
    dataset = row["Dataset"]
    emotion = row['Emotion']
    filename = row['File']
    local_results = []
    try:
        wave, sr = torchaudio.load(path)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            wave = resampler(wave)
        if wave.shape[0] > 1:
            wave = torch.mean(wave, dim=0, keepdim=True)
        if wave.shape[1] < NUM_SAMPLES:
            padding = NUM_SAMPLES - wave.shape[1]
            wave = F.pad(wave, (0, padding))
        if wave.shape[1] > NUM_SAMPLES:
            start = (wave.shape[1] - NUM_SAMPLES) // 2
            end = (wave.shape[1] + NUM_SAMPLES) // 2
            wave = wave[:, start:end]
        wave = normalize_audio(wave)
        save_dir = os.path.join(OUTPUT_DIR, dataset, emotion)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"{filename}.pt")
        torch.save(wave, output_path)
        local_results.append({
            'Dataset': dataset,
            'Emotion': emotion,
            'File': filename
        })
    except Exception as e:
        print(f"Error processing {path}: {e}")
        raise Exception(str(e))
    return local_results

class PreprocessedAudioDataset(Dataset):
    def __init__(self, df, label_encoder, sample_rate=TARGET_SR, super_dir=OUTPUT_DIR, transform=None):
        self.df = df
        self.encoder = label_encoder
        self.main_dir = super_dir
        self.transform = transform or T.MelSpectrogram(
            sample_rate=TARGET_SR, n_fft=1024, hop_length=512, n_mels=64
        )
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        dataset = self.df.iloc[idx]['Dataset']
        emote = self.df.iloc[idx]["Emotion"]
        filename = self.df.iloc[idx]['File']
        tensor_path = os.path.join(self.main_dir, dataset, emote, f"{filename}.pt")
        wave = torch.load(tensor_path)
        wave = (wave - wave.mean()) / wave.std()
        mel = self.transform(wave)
        mel = torch.log1p(mel)
        mel = (mel - mel.mean()) / mel.std()
        label = self.encoder.transform([emote])[0]
        return filename, mel, wave, label

def main():
    global audios
    # Process CREMA-D files
    cremad_files = os.listdir(CREMAD)
    cremad_data = Parallel(n_jobs=-1, verbose=5)(delayed(process_cremad_file)(file) for file in cremad_files)
    audios = pd.concat((audios, pd.DataFrame(cremad_data)), ignore_index=True)

    # Process RAVDESS files
    ravdess_data = process_ravdess_files()
    audios = pd.concat((audios, pd.DataFrame(ravdess_data)), ignore_index=True)

    # Process TESS files
    tess_data = process_tess_files()
    audios = pd.concat((audios, pd.DataFrame(tess_data)), ignore_index=True)

    # Process SAVEE files
    savee_data = process_savee_files()
    audios = pd.concat((audios, pd.DataFrame(savee_data)), ignore_index=True)

    # Process IEMOCAP files
    #iemocap_results = Parallel(n_jobs=-1, verbose=5)(delayed(process_iemocap_session)(IEMOCAP, num) for num in range(1, 6))
    #merged_iemocap = pd.concat(iemocap_results, ignore_index=True)
    #audios = pd.concat([audios, merged_iemocap], ignore_index=True)

    # Verify audio files
    sr_df, invalid_files = get_info(audios)

    # Remove invalid files
    if invalid_files:
        audios.drop(invalid_files, axis=0, inplace=True)
        audios.reset_index(drop=True, inplace=True)

    # Preprocess and save audio tensors
    results = Parallel(n_jobs=-1, verbose=5)(delayed(process_row)(idx, row) for idx, row in audios.iterrows())
    flattened = [item for sublist in results for item in sublist]
    saved_files = pd.DataFrame(flattened)
    print(f"Preprocessing complete! Total chunks: {len(saved_files)}")
    saved_files.to_csv("saved_files.csv", index=False, columns=['Dataset', "Emotion", "File"])

    # Reload dataset and split
    audios = pd.read_csv('saved_files.csv')
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(audios, test_size=0.2, stratify=audios.Emotion)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df.Emotion)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(train_df.Emotion)
    emotes2idxs = {label: idx for idx, label in enumerate(le.classes_)}
    print("Emotion to index mapping:", emotes2idxs)

    # Create datasets
    train_dataset = PreprocessedAudioDataset(df=train_df, label_encoder=le)
    val_dataset = PreprocessedAudioDataset(df=val_df, label_encoder=le)
    test_dataset = PreprocessedAudioDataset(df=test_df, label_encoder=le)

    # Check for NaNs in datasets and remove those samples
    def filter_nan(dataset):
        filtered_data = []
        for file, mel, wave, lbl in dataset:
            if torch.isnan(wave).any():
                print(f"NaN encountered in file: {file} in dataset, removing sample")
            else:
                filtered_data.append((file, mel, wave, lbl))
        return filtered_data

    train_dataset_filtered = filter_nan(train_dataset)
    val_dataset_filtered = filter_nan(val_dataset)
    test_dataset_filtered = filter_nan(test_dataset)

    # Visualize some samples from train, val, test datasets
    for dataset, name in zip([train_dataset, val_dataset, test_dataset], ['Train', 'Validation', 'Test']):
        for file, mel, wave, lbl in dataset:
            print(f"{name} set - Emotion: {le.inverse_transform([lbl])[0]}")
            librosa.display.specshow(librosa.amplitude_to_db(mel[0].numpy()), sr=TARGET_SR, cmap='magma', x_axis='time', y_axis='mel')
            ipd.display(ipd.Audio(wave, rate=TARGET_SR))
            plt.colorbar()
            plt.show()
            break

if __name__ == "__main__":
    main()
