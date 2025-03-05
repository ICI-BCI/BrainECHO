import random
import os
import pickle
import codecs
import re
from tqdm import tqdm
from glob import glob


if __name__ == "__main__":
    train_data = []
    valid_data = []
    test_data = []
    all_files = []
    sent_num = 140

    data_dir = "./datasets/Brennan/split_by_subject"
    for split in ["train", "valid", "test"]:
        path = os.path.join('./datasets/Brennan/split_by_sents', split)
        os.makedirs(path, exist_ok=True)

    for split in ["train", "valid", "test"]:
        data_path = os.path.join(data_dir, split)
        files = os.listdir(data_path)
        files.sort(key=lambda f: int(re.match("sub-(\d+)-", f).group(1)))

        for i in range(len(files) // sent_num):
            sub_files = files[i * sent_num : (i + 1) * sent_num]
            random.shuffle(sub_files)
            train_files = sub_files[: int(sent_num * 0.8)]
            valid_files = sub_files[int(sent_num * 0.8) : int(sent_num * 0.9)]
            test_files = sub_files[int(sent_num * 0.9) :]

            for file in tqdm(train_files):
                with open(os.path.join(data_path, file), "rb") as f:
                    data = pickle.load(f)
                    output_name = f"./datasets/Brennan/split_by_sents/train/sub-{data['subject']}-sent-{data['sent_id']}.pickle"
                    with codecs.open(output_name, "wb") as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            for file in tqdm(valid_files):
                with open(os.path.join(data_path, file), "rb") as f:
                    data = pickle.load(f)
                    output_name = f"./datasets/Brennan/split_by_sents/valid/sub-{data['subject']}-sent-{data['sent_id']}.pickle"
                    with codecs.open(output_name, "wb") as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            for file in tqdm(test_files):
                with open(os.path.join(data_path, file), "rb") as f:
                    data = pickle.load(f)
                    output_name = f"./datasets/Brennan/split_by_sents/test/sub-{data['subject']}-sent-{data['sent_id']}.pickle"
                    with codecs.open(output_name, "wb") as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
