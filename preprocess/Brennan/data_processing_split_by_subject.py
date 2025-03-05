import random
import os
import pickle
import codecs
from tqdm import tqdm
from glob import glob


if __name__ == "__main__":
    train_data = []
    valid_data = []
    test_data = []
    sub_ids = []
    for file in glob("/home/szx/dataset/Brennan/S*.mat"):
        sub_ids.append(int(file[1:3]))
    random.shuffle(sub_ids)
    train_sub_ids = sub_ids[:27]
    valid_sub_ids = sub_ids[27:30]
    test_sub_ids = sub_ids[30:]

    data_dir = "./datasets/Brennan/split_by_shuffle"
    for split in ["train", "valid", "test"]:
        data_path = os.path.join(data_dir, split)
        files = os.listdir(data_path)

        for file in tqdm(files, ncols=80):
            with open(os.path.join(data_path, file), "rb") as f:
                data = pickle.load(f)
                if data["subject"] in train_sub_ids:
                    train_data.append(data)
                elif data["subject"] in valid_sub_ids:
                    valid_data.append(data)
                elif data["subject"] in test_sub_ids:
                    test_data.append(data)
                else:
                    print("unknown sub id")

    for split in ["train", "valid", "test"]:
        path = f"./datasets/Brennan/split_by_subject/{split}"
        os.makedirs(path, exist_ok=True)

    for eeg_data in train_data:
        output_name = f"./datasets/Brennan/split_by_subject/train/sub-{eeg_data['subject']}-sent-{eeg_data['sent_id']}.pickle"
        with codecs.open(output_name, "wb") as handle:
            pickle.dump(eeg_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for eeg_data in valid_data:
        output_name = f"./datasets/Brennan/split_by_subject/valid/sub-{eeg_data['subject']}-sent-{eeg_data['sent_id']}.pickle"
        with codecs.open(output_name, "wb") as handle:
            pickle.dump(eeg_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for eeg_data in test_data:
        output_name = f"./datasets/Brennan/split_by_subject/test/sub-{eeg_data['subject']}-sent-{eeg_data['sent_id']}.pickle"
        with codecs.open(output_name, "wb") as handle:
            pickle.dump(eeg_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
