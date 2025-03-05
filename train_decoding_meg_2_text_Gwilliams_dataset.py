import os
import argparse
from utils.utils import *
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
import copy
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from model import BrainECHO
from optim_new import *
from einops.einops import rearrange
from peft import AdaLoraConfig
from reader import MEGDataset, meg_collate_fn
import math


"""
follow by the article "Open Vocabulary Electroencephalography-to-Text Decoding and Zero-Shot Sentiment Classification"
"""


def train_model(
    dataloaders,
    device,
    model,
    optimizer,
    scheduler,
    early_stopping,
    num_epochs=25,
    phases=("train", "dev"),
    checkpoint_path_best="./checkpoints/decoding/best/temp_decoding.pt",
):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    start_time = time.time()
    best_valid_loss = None
    idx = 0
    step = 0
    stop = False

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        logger.info("Epoch {}/{}\n".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == "train":
                model.train()  # Set model to training mode
            elif phase == "dev":
                model.eval()  # Set model to evaluate mode
            elif phase == "test":
                model.eval()

            running_loss = 0.0
            mel_running_loss = 0.0

            # Iterate over data.
            for data in tqdm(dataloaders[phase], ncols=80):
                rawEEG_batch = data["input_features"][..., :4800]
                rawEEG_batch = rawEEG_batch[..., ::2].to(device).float()
                b = rawEEG_batch.shape[0]
                if split_mel:
                    rawEEG_batch = rearrange(rawEEG_batch, "b c (p t)->(b p) c t", t=mel_len * 2)

                if model.task == "task1":
                    target_ids_batch = None
                    target_mel_feature_batch = data["log_mel_features"].to(device)
                    target_mel_feature_batch = F.interpolate(target_mel_feature_batch.unsqueeze(1), (80, 1200), mode="bilinear").squeeze(1)
                    # split mel spectrogram into segments
                    if split_mel:
                        target_mel_feature_batch[
                            target_mel_feature_batch
                            == target_mel_feature_batch.reshape(b, -1).min(1).values.reshape(b, 1, 1)
                        ] = -1
                        target_mel_feature_batch = rearrange(
                            target_mel_feature_batch, "b d (p t)->(b p) d t", t=mel_len
                        )
                    if random.random() < 0.05:
                        noise = torch.randn_like(rawEEG_batch[:1])
                        rawEEG_batch = torch.cat([rawEEG_batch, noise])
                        target_mel_feature_batch = torch.cat([target_mel_feature_batch, noise_mel])
                else:
                    target_ids_batch = data["labels"].to(device)
                    target_mel_feature_batch = None

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == "train":
                    out = model(rawEEG_batch, target_ids_batch, target_mel_feature_batch)
                else:
                    with torch.no_grad():
                        out = model(rawEEG_batch, target_ids_batch, target_mel_feature_batch)
                loss = out.loss
                if model.task == "task1":
                    mel_loss = out.mel_loss

                """calculate loss"""
                # backward + optimize only if in training phase
                if phase == "train":
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()
                    if model.task == "task2":
                        model.pretrained.model.encoder.update_and_allocate(step)
                        step += 1

                # statistics
                running_loss += loss.item() * rawEEG_batch.size()[0]  # batch loss
                if model.task == "task1":
                    mel_running_loss += mel_loss.item() * rawEEG_batch.size()[0]

                idx += 1

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print("{} Loss: {:.4f}".format(phase, epoch_loss))
            logger.info("{} Loss: {:.4f}".format(phase, epoch_loss))

            if model.task == "task1":
                epoch_mel_loss = mel_running_loss / dataset_sizes[phase]
                print("{} mel Loss: {:.4f}".format(phase, epoch_mel_loss))
                logger.info("{} mel Loss: {:.4f}".format(phase, epoch_mel_loss))

            if phase == "dev":
                save = False
                if best_valid_loss is None or best_valid_loss[1] > epoch_loss:
                    best_valid_loss = (epoch, epoch_loss)
                    save = True

                if early_stopping.early_stop(epoch_loss):
                    print("Early stopping")
                    stop = True

                if save:
                    """save checkpoint"""
                    if model.task == "task1":
                        torch.save(model.state_dict(), checkpoint_path_best)
                    elif model.task == "task2":
                        temp_model = copy.deepcopy(model)
                        temp_model.pretrained.model.encoder = temp_model.pretrained.model.encoder.merge_and_unload()
                        torch.save(temp_model.state_dict(), checkpoint_path_best)
                    print(f"update best on dev checkpoint: {checkpoint_path_best}")
                    logger.info("update best on dev checkpoint:{}".format(checkpoint_path_best))

        print()
        if stop:
            stop = False
            break

    if model.task == "task2":
        print("Finish training, please take a look.")
        print("The best loss {} in epoch {},".format(best_valid_loss[1], best_valid_loss[0]))
        logger.info("\n" "The best loss {} in epoch {}".format(best_valid_loss[1], best_valid_loss[0]))

    end_time = time.time()
    run_time = math.ceil(end_time - start_time)
    print(f"time consuming: {datetime.timedelta(seconds = run_time)}")
    logger.info(f"time consuming: {datetime.timedelta(seconds = run_time)}")

    return


def show_require_grad_layers(model):
    print()
    print("require_grad layers:")
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


def save_require_grad_layers_to_txt(model, filename):
    with open(filename, "w") as file:
        file.write("require_grad layers:\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                file.write(f"{name}\n")
    print(f"Saved require_grad layers to '{filename}'.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MEG-Text")
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument("--save_path", help="path to save checkpoint", required=True)
    parser.add_argument("-s", "--subject", default=None, help="subject")
    parser.add_argument("--split", default="split1", help="dataset split setting")
    parser.add_argument("--split_mel", action="store_true", default=False, help="split the mel to n s window")
    parser.add_argument("--split_len", default=4, help="mel split window")
    parser.add_argument(
        "--latent_loss", default="mse", choices=["mse", "cos_sim", "clip_loss"], help="latent loss criterion"
    )
    parser.add_argument("--cb_size", default=2048, help="codebook size")
    parser.add_argument("-r", "--ratio", default=4, help="downsampling ratio")
    
    """ parse args """
    args = vars(parser.parse_args())
    subject = args["subject"]
    subject = int(subject) if subject is not None else subject
    split = args["split"]
    split_mel = args["split_mel"]
    latent_loss = args["latent_loss"]
    split_len = int(args["split_len"]) * 100
    codebook_size = int(args["cb_size"])
    r = int(args["ratio"])
    mel_len = 1200 if not split_mel else split_len

    save_path = args["save_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    """ config param"""
    args = read_configuration(args["config"])
    vqvae_ckpt = args["checkpoint"]
    model_name = args["model_name"]

    init_logger(args)
    logger = getLogger()
    print(f"[INFO]using model: {model_name}")

    task_name = args["task_name"]

    """ set random seeds """
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.benchmark = True

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

    print()

    processor = WhisperProcessor.from_pretrained("/home/szx/eeg2text/models/huggingface/whisper-base.en")

    """ dataset """
    train_set = MEGDataset(
        f"/home/szx/dataset/meg-masc-main/preprocess/{split}/train.jsonl",
        processor=processor,
        modal="eeg",
        modal_ch=208,
        mode="train",
        sample_rate=200,
        orig_sample_rate=200,
        language="English",
        filter_dataset=False,
        timestamps=False,
        combine_sentences=False,
        split_sentences=False,
        min_duration=0.5,
        max_duration=24,
        subj=subject,
        augment_config_path="",
    )
    valid_set = MEGDataset(
        f"/home/szx/dataset/meg-masc-main/preprocess/{split}/val.jsonl",
        processor=processor,
        modal="eeg",
        modal_ch=208,
        mode="val",
        sample_rate=200,
        orig_sample_rate=200,
        language="English",
        filter_dataset=False,
        timestamps=False,
        combine_sentences=False,
        split_sentences=False,
        min_duration=0.5,
        max_duration=24,
        subj=subject,
        augment_config_path="",
    )
    dataset_sizes = {"train": len(train_set), "dev": len(valid_set)}
    print("[INFO]train_set size: ", len(train_set))
    print("[INFO]dev_set size: ", len(valid_set))
    if len(train_set) == 0:
        sys.exit()

    """ set up model """
    model = BrainECHO(
        None,
        in_chan=208,
        d_model=256,
        ffn_dim=1024,
        num_layers=4,
        mel_len=mel_len,
        mel_interp=True,
        use_mlp=True,
        codebook_size=codebook_size,
        r=r,
        split_mel=split_mel,
    )
    model.mel_vqvae.load_state_dict(
        torch.load(vqvae_ckpt)
    )

    model.to(device)

    """ training loop """

    ######################################################
    """ stage two training: brain-audio alignment """
    ######################################################

    print("=== start task1 ... ===")
    model.task = "task1"

    """ training param """
    num_epoch_finetune = args["num_epoch_finetune"][0]
    lr_finetune = args["lr_finetune"]
    args["lr_finetune"] = lr_finetune[0]
    batch_size = args["batch_size"][0]
    T = args["T_max"][0]
    logger.info("lr_finetune:{}".format(args["lr_finetune"]))

    """ dataloader """
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=meg_collate_fn, pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=meg_collate_fn, pin_memory=True
    )
    dataloaders = {"train": train_dataloader, "dev": valid_dataloader}

    """ set up optimizer and scheduler """
    early_stopping = EarlyStopper(patience=4, min_delta=0.0)
    optimizer = build_optimizer(args, model, mode="finetune")
    logger.info("T:{}".format(T))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)

    """ checkpoint save """
    save_name = f"b{batch_size}_{num_epoch_finetune}_{args['lr_finetune']}_alignment"
    if subject is not None:
        save_name = f"sub{subject}_b{batch_size}_{num_epoch_finetune}_{args['lr_finetune']}_alignment"
    output_checkpoint_name_best = save_path + f"/{save_name}"
    checkpoint_path_best = f"{output_checkpoint_name_best}.pt"

    # add noise samples to enhance robustness
    noise_audio = [0] * int(mel_len / 100 * 16000)
    noise_mel = processor(noise_audio, sampling_rate=16000, return_tensors="pt").input_features
    noise_mel = noise_mel[..., :mel_len].to(device)
    if split_mel:
        noise_mel = torch.clip(noise_mel, min=-1)

    show_require_grad_layers(model)
    train_model(
        dataloaders,
        device,
        model,
        optimizer,
        scheduler,
        early_stopping,
        num_epochs=num_epoch_finetune,
        checkpoint_path_best=checkpoint_path_best,
    )
    print()

    ######################################################
    """ stage three training: Whisper finetuning """
    ######################################################

    print("=== start task2 ... ===")
    model.task = "task2"

    del noise_mel

    """ training param """
    num_epoch_finetune = args["num_epoch_finetune"][1]
    args["lr_finetune"] = lr_finetune[1]
    batch_size = args["batch_size"][1]
    T = args["T_max"][1]
    logger.info("lr_finetune:{}".format(args["lr_finetune"]))

    """ dataloader """
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=meg_collate_fn, pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=meg_collate_fn, pin_memory=True
    )
    dataloaders = {"train": train_dataloader, "dev": valid_dataloader}

    """ set up optimizer and scheduler """
    early_stopping = EarlyStopper(patience=4, min_delta=0.0)
    optimizer = build_optimizer(args, model, mode="finetune")
    logger.info("T:{}".format(T))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)

    """ checkpoint save """
    save_name = f"b{batch_size}_{num_epoch_finetune}_{args['lr_finetune']}_finetune"
    if subject != "all":
        save_name = f"sub{subject}_b{batch_size}_{num_epoch_finetune}_{args['lr_finetune']}_finetune"
    output_checkpoint_name_best = save_path + f"/{save_name}"
    checkpoint_path_best = f"{output_checkpoint_name_best}.pt"

    """ add pretrained layers """
    pretrained = WhisperForConditionalGeneration.from_pretrained("/home/szx/eeg2text/models/huggingface/whisper-base.en").to(device)
    for p in model.parameters():
        p.requires_grad = False
    for p in pretrained.parameters():
        p.requires_grad = False
    config = AdaLoraConfig(
        init_r=12,
        target_r=4,
        tinit=200,
        tfinal=1000,
        total_step=math.ceil(dataset_sizes["train"] / batch_size) * num_epoch_finetune,
        deltaT=10,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    model.add_adalora_pretrained_model(pretrained, config)

    show_require_grad_layers(model)
    train_model(
        dataloaders,
        device,
        model,
        optimizer,
        scheduler,
        early_stopping,
        num_epochs=num_epoch_finetune,
        checkpoint_path_best=checkpoint_path_best,
    )
