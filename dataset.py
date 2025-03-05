from torch.utils.data import Dataset
import os
import pickle
import torch


# 如果要改input_embeddings变为 normalized_input_embeddings 在这里
class EEG_dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(os.path.join(self.path, file), "rb") as handle:
            input_sample = pickle.load(handle)
        return (
            input_sample["input_embeddings"],
            input_sample["input_attn_mask"],
            input_sample["input_attn_mask_invert"],
            input_sample["target_ids"],
            input_sample["target_mask"],
            input_sample["target_string"],
        )


class EEG_dataset_whisper(Dataset):
    def __init__(self, path, subject="all"):
        """Brennan EEG dataset

        Args:
            path (str): path to dataset
            subject (str, optional): subject ID. Defaults to "all".
        """        
        self.path = path
        self.files = os.listdir(path)
        self.files.sort(key=lambda x: (int(x.split("-")[1]), int(x.split("-")[3].replace(".pickle", ""))))
        if subject != "all":
            self.files = [file for file in self.files if f"sub{subject}-" in file or f"sub-{subject}-" in file]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(os.path.join(self.path, file), "rb") as handle:
            input_sample = pickle.load(handle)
        return (
            input_sample["rawEEG"],
            input_sample["input_attn_mask"],
            input_sample["target_ids"],
            input_sample["target_mask"],
            input_sample["sent_id"],
            input_sample["target_string"],
        )


class MEG_dataset_whisper(Dataset):
    def __init__(self, path, split="train"):
        """GWilliams EEG dataset (unused)

        Args:
            path (str): path to dataset
            split (str, optional): train, valid or test. Defaults to "train".
        """        
        path = os.path.join(path, split)
        files = os.listdir(path)
        self.input_samples = []
        for file in files:
            with open(os.path.join(path, file), "rb") as handle:
                input_sample = pickle.load(handle)
                self.input_samples.extend(input_sample)

    def __len__(self):
        return len(self.input_samples)

    def __getitem__(self, idx):
        input_sample = self.input_samples[idx]
        return (
            input_sample["MEG_start"],
            input_sample["MEG_end"],
            input_sample["target_ids"],
            input_sample["target_mask"],
            input_sample["audio_file"],
            input_sample["audio_start"],
            input_sample["audio_end"],
            input_sample["target_string"],
            input_sample["subject"],
            input_sample["session"],
            input_sample["task"],
        )
