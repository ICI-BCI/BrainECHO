import librosa
import mne
import math
import torch
import numpy as np
import codecs
import pickle
import os
import random
from tqdm import tqdm
from sklearn.preprocessing import robust_scale
from scipy.io import loadmat
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
)


def get_input_sample(raw, subject=1):
    max_seq_len = 45
    data = []

    raw_data = raw.get_data()
    raw_data = robust_scale(raw_data.T).T
    raw_data = np.clip(raw_data, -10, 10)
    raw_data = raw_data / np.max(np.abs(raw_data))
    raw_data = torch.tensor(raw_data)
    if raw_data.shape[0] < 62:
        raw_data = torch.cat(
            [raw_data, torch.zeros(62 - raw_data.shape[0], raw_data.shape[1])]
        )

    max_sent_id = len(all_sents)

    for sent_id in range(max_sent_id):
        input_sample = {"subject": subject}

        start_time = audio_split[sent_id][0]
        start_time = int(start_time * resample_sr)
        end_time = audio_split[sent_id][1]
        end_time = math.ceil(end_time * resample_sr)
        eeg_len = end_time - start_time

        padded_data = torch.cat(
            [raw_data[:, start_time:end_time], torch.zeros(62, 2400 - eeg_len)], 1
        )
        input_sample["rawEEG"] = padded_data
        input_sample["target_string"] = all_sents[sent_id]
        target_tokenized = tokenizer(
            input_sample["target_string"],
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_sample["target_ids"] = target_tokenized["input_ids"][0]
        input_sample["target_mask"] = target_tokenized["attention_mask"][0]
        input_sample["sent_id"] = sent_id
        input_sample["input_attn_mask"] = torch.zeros(2400).int()  # 0 is masked out
        input_sample["input_attn_mask"][:eeg_len] = torch.ones(
            eeg_len
        ).int()  # 1 is not masked

        data.append(input_sample)

    return data


if __name__ == "__main__":
    with open(
        "/home/szx/dataset/Brennan/preprocess/audio/transcription/DownTheRabbitHoleFinal_SoundFile_16kHz.json",
        "r",
        encoding="utf-8",
    ) as f:
        meta = eval(f.read())

    audio, sr = librosa.load(
        "/home/szx/dataset/Brennan/preprocess/audio/wav16/DownTheRabbitHoleFinal_SoundFile_16kHz.wav", sr=16000
    )

    # load tokenizer and processor
    processor = WhisperProcessor.from_pretrained(
        r"/home/szx/eeg2text/models/huggingface/whisper-base.en"
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        r"/home/szx/eeg2text/models/huggingface/whisper-base.en"
    )

    all_sents = []
    audio_split = []
    all_audio = []
    for sent_meta in meta["segments"]:
        sent = sent_meta["text"]
        if not sent.startswith(" "):
            sent = f" {sent}"
        all_sents.append(sent)

        audio_start = sent_meta["start"]
        audio_end = sent_meta["end"]
        audio_split.append((audio_start, audio_end))
        all_audio.append(audio[int(audio_start * 16000) : math.ceil(audio_end * 16000)])

    features = processor(
        all_audio, sampling_rate=16000, return_tensors="pt"
    ).input_features
    print(features.shape)

    with open("./datasets/Brennan/all_sents.pickle", "wb") as f:
        pickle.dump(all_sents, f)
    with open("./datasets/Brennan/log_mel_features.pickle", "wb") as f:
        pickle.dump(features[:, :, :1200], f)

    del all_audio
    del features

    resample_sr = 200
    for subject in tqdm(range(1, 50), position=0):
        file_name = "/home/szx/dataset/Brennan/S%s.mat" % str(subject).rjust(2, "0")
        print(f"---------------sub{subject}---------------")
        print()
        if not os.path.exists(file_name):
            continue

        data = loadmat(file_name)
        raw = data["raw"][0][0][3][0][0]
        info = mne.create_info(
            ch_names=[f"EEG{i}" for i in range(1, raw.shape[0] + 1)],
            sfreq=500,
            ch_types=["eeg"] * raw.shape[0],
        )
        raw = mne.io.RawArray(raw, info)
        raw.load_data()
        raw.notch_filter(freqs=(60,))
        raw.filter(0.5, 99.0, n_jobs=1)
        raw.resample(resample_sr)

        data = get_input_sample(raw, subject)

        del raw

        seq_num = len(data)
        random.shuffle(data)
        train_data = data[: int(seq_num * 0.8)]
        valid_data = data[int(seq_num * 0.8) : int(seq_num * 0.9)]
        test_data = data[int(seq_num * 0.9) :]

        for eeg_data in train_data:
            output_name = f"./datasets/Brennan/split_by_shuffle/train/sub-{eeg_data['subject']}-sent-{eeg_data['sent_id']}.pickle"
            with codecs.open(output_name, "wb") as handle:
                pickle.dump(eeg_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for eeg_data in valid_data:
            output_name = f"./datasets/Brennan/split_by_shuffle/valid/sub-{eeg_data['subject']}-sent-{eeg_data['sent_id']}.pickle"
            with codecs.open(output_name, "wb") as handle:
                pickle.dump(eeg_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for eeg_data in test_data:
            output_name = f"./datasets/Brennan/split_by_shuffle/test/sub-{eeg_data['subject']}-sent-{eeg_data['sent_id']}.pickle"
            with codecs.open(output_name, "wb") as handle:
                pickle.dump(eeg_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
