import os
import argparse
from utils.utils import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from model import BrainECHO
from optim_new import *
from metrics import compute_metrics
from einops import rearrange
from reader import MEGDataset, meg_collate_fn
import math


"""
follow by the article "Open Vocabulary Electroencephalography-to-Text Decoding and Zero-Shot Sentiment Classification"
"""


def eval_model(dataloaders, device, tokenizer, model, output_all_results_path="./results/temp.txt"):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    if is_teacher_forcing:
        model.eval()  # Set model to evaluate mode
        model.task = "task2"
        target_string_list = []
        pred_string_list = []
        idx = 0
        start_time = time.time()
        compute_cb_usage = not random_select
        codebook_usage = torch.zeros(codebook_size).to(device)

        with open(output_all_results_path, "w") as f:
            for data in tqdm(dataloaders["test"], ncols=80):
                # load in batch
                rawEEG_batch = data["input_features"][..., :4800]
                rawEEG_batch = rawEEG_batch[..., ::2].to(device).float()
                # split mel spectrogram into segments
                if split_mel:
                    rawEEG_batch = rearrange(rawEEG_batch, "b c (p t)->(b p) c t", t=mel_len*2)
                if input_noise:
                    rawEEG_batch = torch.randn_like(rawEEG_batch)

                target_ids_batch = data["labels"].to(device)

                target_string = tokenizer.batch_decode(target_ids_batch, skip_special_tokens=True)

                # add to list for later calculate bleu metric
                target_string_list.extend(target_string)

                with torch.no_grad():
                    output = model(rawEEG_batch, target_ids_batch, None)
                logits = output.logits[:, 1:]  # b*l*50265

                probs = logits.softmax(dim=-1)
                values, predictions = probs.topk(1)
                predictions = torch.squeeze(predictions)

                if compute_cb_usage:
                    indices = output.indices
                    indices = indices.unique()
                    codebook_usage[indices] = 1

                predicted_string = tokenizer.batch_decode(predictions, skip_special_tokens=True)

                for i in range(len(target_string)):
                    f.write(f"target string: {target_string[i]}\n")
                    f.write(f"predicted string: {predicted_string[i]}\n")
                    f.write(f"################################################\n\n\n")
                    idx += 1

                pred_string_list.extend(predicted_string)
                # print('################################################')
                # print()

        end_time = time.time()
        print(f"inference time: {end_time - start_time}s")
        logger.info(f"inference time: {end_time - start_time}s")

        """ calculate corpus bleu score and rouge score """
        result = compute_metrics(pred_string_list, target_string_list)
        for k, v in result.items():
            print(f"{k}: {v}")
        logger.info(result)

        if compute_cb_usage:
            print(f"codebook usage: {codebook_usage.sum().item()/codebook_size}")
            logger.info(f"codebook usage: {codebook_usage.sum().item()/codebook_size}")

    else:
        model.eval()  # Set model to evaluate mode

        target_string_list = []
        pred_string_list = []
        idx = 0
        start_time = time.time()
        compute_cb_usage = not random_select
        codebook_usage = torch.zeros(codebook_size).to(device)

        with open(output_all_results_path, "w") as f:
            for data in tqdm(dataloaders["test"], ncols=80):
                # load in batch
                rawEEG_batch = data["input_features"][..., :4800]
                rawEEG_batch = rawEEG_batch[..., ::2].to(device).float()
                # split mel spectrogram into segments
                if split_mel:
                    rawEEG_batch = rearrange(rawEEG_batch, "b c (p t)->(b p) c t", t=mel_len*2)
                if input_noise:
                    rawEEG_batch = torch.randn_like(rawEEG_batch)

                target_ids_batch = data["labels"].to(device)

                target_string = tokenizer.batch_decode(target_ids_batch, skip_special_tokens=True)

                # add to list for later calculate bleu metric
                target_string_list.extend(target_string)

                if random_select:
                    all_sent_ids = set(range(len(all_sents)))
                    all_sent_ids = list(all_sent_ids)
                    random_ids = random.choices(all_sent_ids, k=rawEEG_batch.shape[0])
                    predicted_string = [all_sents[random_id] for random_id in random_ids]
                else:
                    output = model.generate(
                        rawEEG_batch,
                        target_ids_batch,
                        None,
                        max_length=256,
                        num_beams=5,
                        do_sample=False,
                        repetition_penalty=5.0,
                        no_repeat_ngram_size=2
                    )
                    predictions = output.predictions
                    predicted_string = tokenizer.batch_decode(predictions, skip_special_tokens=True)

                if compute_cb_usage:
                    indices = output.indices
                    indices = indices.unique()
                    codebook_usage[indices] = 1

                for i in range(len(target_string)):
                    f.write(f"target string: {target_string[i]}\n")
                    f.write(f"predicted string: {predicted_string[i]}\n")
                    f.write(f"################################################\n\n\n")

                pred_string_list.extend(predicted_string)
                # print('################################################')
                # print()
                idx += 1

        end_time = time.time()
        run_time = math.ceil(end_time - start_time)
        print(f"inference time: {datetime.timedelta(seconds = run_time)}")
        logger.info(f"inference time: {datetime.timedelta(seconds = run_time)}")

        """ calculate corpus bleu score and rouge score """
        result = compute_metrics(pred_string_list, target_string_list)
        for k, v in result.items():
            print(f"{k}: {v}")
        logger.info(result)

        if compute_cb_usage:
            print(f"codebook usage: {codebook_usage.sum().item()/codebook_size}")
            logger.info(f"codebook usage: {codebook_usage.sum().item()/codebook_size}")


def show_require_grad_layers(model):
    print()
    print(" require_grad layers:")
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(" ", name)


def save_require_grad_layers_to_txt(model, filename):
    with open(filename, "w") as file:
        file.write("require_grad layers:\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                file.write(f"{name}\n")
    print(f"Saved require_grad layers to '{filename}'.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="EEG-Text")
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument("--text_rusults", help="path to decoding result", required=True)
    parser.add_argument("--eeg2text_checkpoint", help="path to checkpoint", required=True)
    parser.add_argument("-s", "--subject", default="all", help="subject")
    parser.add_argument("--split", default="split1", help="dataset split setting")
    parser.add_argument("--split_mel", action="store_true", default=False, help="split the mel to 4s window")
    parser.add_argument("--split_len", default=4, help="mel split window")
    parser.add_argument("--cb_size", default=2048, help="codebook size")
    parser.add_argument("-r", "--ratio", default=4, help="downsample ratio")
    parser.add_argument("--noise", action="store_true", default=False, help="input noise")
    parser.add_argument(
        "--random_select", action="store_true", default=False, help="random select sentences for evaluation"
    )
    parser.add_argument("--tf", action="store_true", default=False, help="use teacher forcing to decode")
    parser.add_argument("--wo_ft", action="store_true", default=False, help="don't use whisper fine-tuning")
    
    """ parse args """
    args = vars(parser.parse_args())
    subject = args["subject"]
    split = args["split"]
    split_mel = args["split_mel"]
    split_len = int(args["split_len"]) * 100
    codebook_size = int(args["cb_size"])
    r = int(args["ratio"])
    input_noise = args["noise"]
    random_select = args["random_select"]
    is_teacher_forcing = args["tf"]
    wo_ft = args["wo_ft"]
    mel_len = 1200 if not split_mel else split_len

    eeg2text_checkpoint = args["eeg2text_checkpoint"]
    print(eeg2text_checkpoint)

    save_name = args["text_rusults"]
    os.makedirs("result", exist_ok=True)
    output_all_results_path = f"./result/{save_name}.txt"

    """ config param"""
    args = read_configuration(args["config"])
    batch_size = args["batch_size"]
    model_name = args["model_name"]
    task_name = args["task_name"]

    init_logger(args)
    logger = getLogger()
    logger.info("dataset_path:{}".format(args["dataset_path"]))

    print(f"[INFO]using model: {model_name}")

    """ set random seeds """
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    """ set up device """
    # use cuda
    if torch.cuda.is_available():
        # dev = "cuda:3"
        dev = args["cuda"]
    else:
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    device = torch.device(dev)
    print(f"[INFO]using device {dev}")
    print()

    """save config"""

    tokenizer = WhisperTokenizer.from_pretrained("/home/szx/eeg2text/models/huggingface/whisper-base.en")
    processor = WhisperProcessor.from_pretrained("/home/szx/eeg2text/models/huggingface/whisper-base.en")

    if random_select:
        with open(args["dataset_path"] + "all_sents.pickle", 'rb') as f:
            all_sents = pickle.load(f)

    """ dataset """
    test_set = MEGDataset(
        f"/data2/meg-masc-main/preprocess/{split}/test.jsonl",
        processor=processor,
        modal="eeg",
        modal_ch=208,
        mode="test",
        sample_rate=200,
        orig_sample_rate=200,
        language="English",
        filter_dataset=False,
        timestamps=False,
        combine_sentences=False,
        split_sentences=False,
        min_duration=0.5,
        max_duration=24,
        augment_config_path="",
    )
    dataset_sizes = {"test": len(test_set)}
    print("[INFO]test_set size: ", len(test_set))

    """ dataloader """
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=meg_collate_fn, pin_memory=True)
    dataloaders = {"test": test_dataloader}

    """ set up model """
    pretrained = WhisperForConditionalGeneration.from_pretrained("/home/szx/eeg2text/models/huggingface/whisper-base.en")
    model = BrainECHO(
        None,
        in_chan=208,
        d_model=256,
        ffn_dim=1024,
        num_layers=4,
        mel_len=mel_len,
        codebook_size=codebook_size,
        mel_interp=True,
        use_mlp=True
    )
    if not wo_ft:
        model.pretrained = pretrained

    model.to(device)

    model.load_state_dict(torch.load(eeg2text_checkpoint), strict=True)
    if wo_ft:
        model.pretrained = pretrained.to(device)

    model.task = "task2"
    eval_model(dataloaders, device, tokenizer, model, output_all_results_path=output_all_results_path)
