import os
import argparse
from utils.utils import *
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import time
from tqdm import tqdm
from transformers import WhisperProcessor
from model import MelVQVAE
from optim_new import *
from einops.einops import rearrange
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
    phases=("train",),
    checkpoint_path_best="./checkpoints/decoding/best/temp_decoding.pt",
):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    start_time = time.time()
    best_valid_loss = None
    idx = 0
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

            # Iterate over data
            for data in tqdm(dataloaders[phase], ncols=80):
                b = data[0].shape[0]

                target_mel_feature_batch = data[0].to(device)
                # split mel spectrogram into segments
                if split_mel:
                    target_mel_feature_batch[
                        target_mel_feature_batch
                        == target_mel_feature_batch.reshape(b, -1).min(1).values.reshape(b, 1, 1)
                    ] = -1
                    target_mel_feature_batch = rearrange(target_mel_feature_batch, "b d (p t)->(b p) d t", t=mel_len)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == "train":
                    out = model(target_mel_feature_batch, loss="mse")
                else:
                    with torch.no_grad():
                        out = model(target_mel_feature_batch, loss="mse")
                loss = out.loss
                mel_loss = out.mel_loss

                """calculate loss"""
                # NOTE: my criterion not used
                #
                # # backward + optimize only if in training phase
                if phase == "train":
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * b  # batch loss
                mel_running_loss += mel_loss.item() * b

                idx += 1

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print("{} Loss: {:.4f}".format(phase, epoch_loss))
            logger.info("{} Loss: {:.4f}".format(phase, epoch_loss))

            epoch_mel_loss = mel_running_loss / dataset_sizes[phase]
            print("{} mel Loss: {:.4f}".format(phase, epoch_mel_loss))
            logger.info("{} mel Loss: {:.4f}".format(phase, epoch_mel_loss))

            if phase == "train":
                save = False
                if best_valid_loss is None or best_valid_loss[1] > epoch_loss:
                    best_valid_loss = (epoch, epoch_loss)
                    save = True

                if early_stopping.early_stop(epoch_loss):
                    print("Early stopping")
                    stop = True

                if save:
                    """save checkpoint"""
                    torch.save(model.state_dict(), checkpoint_path_best)
                    print(f"update best on dev checkpoint: {checkpoint_path_best}")
                    logger.info("update best on dev checkpoint:{}".format(checkpoint_path_best))

        print()
        if stop:
            stop = False
            break

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

    parser = argparse.ArgumentParser(description="Mel-Autoencoding")
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument("--split_mel", action="store_true", default=False, help="split the mel to n s window")
    parser.add_argument("--split_len", default=4, help="mel split window")
    parser.add_argument("--cb_size", default=2048, help="codebook size")
    parser.add_argument("-r", "--ratio", default=4, help="downsample ratio")
    parser.add_argument(
        "--dataset", default="both", choices=["Brennan", "gwilliams", "both"], help="dataset selected to pretrain"
    )

    """ parse args """
    args = vars(parser.parse_args())
    split_mel = args["split_mel"]
    dataset = args["dataset"]
    split_len = args["split_len"] * 100
    codebook_size = int(args["cb_size"])
    downsample_ratio = int(args["ratio"])
    mel_len = 1200 if not split_mel else split_len

    """ config param """
    args = read_configuration(args["config"])
    num_epoch_finetune = args["num_epoch_finetune"]
    lr_finetune = args["lr_finetune"]
    batch_size = args["batch_size"]
    model_name = args["model_name"]

    init_logger(args)
    logger = getLogger()
    logger.info("dataset_path:{}".format(args["dataset_path"]))
    logger.info("lr_finetune:{}".format(lr_finetune))
    print(f"[INFO]using model: {model_name}")

    task_name = args["task_name"]

    save_path = args["save_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    """ checkpoint save """
    save_name = f"{model_name}_b{batch_size}_{num_epoch_finetune}_{lr_finetune}_autoencode"
    output_checkpoint_name_best = save_path + f"/{save_name}"
    checkpoint_path_best = f"{output_checkpoint_name_best}.pt"

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

    """ prepare log mel """
    with open("datasets/Brennan/log_mel_features.pickle", "rb") as f:
        Brennan_log_mel_features = pickle.load(f)  # (140, 80, 1200)
    with open("datasets/meg/log_mel_features.pickle", "rb") as f:
        gwilliams_log_mel_features = pickle.load(f)  # (661, 80, 2400)
        gwilliams_log_mel_features = F.interpolate(
            gwilliams_log_mel_features.unsqueeze(1), (80, 1200), mode="bilinear"
        ).squeeze(
            1
        )  # (661, 80, 1200)
    if dataset == "Brennan":
        log_mel_features = Brennan_log_mel_features
    elif dataset == "gwilliams":
        log_mel_features = gwilliams_log_mel_features
    elif dataset == "both":
        log_mel_features = torch.cat([Brennan_log_mel_features, gwilliams_log_mel_features])

    # add noise samples to enhance robustness
    noise_audio = [0] * (12 * 16000)
    # processor = WhisperProcessor.from_pretrained("models/huggingface/whisper-base.en")
    processor = WhisperProcessor.from_pretrained("/home/szx/eeg2text/models/huggingface/whisper-base.en")
    noise_mel = processor(noise_audio, sampling_rate=16000, return_tensors="pt").input_features
    noise_mel = noise_mel[..., :1200]
    log_mel_features = torch.cat([log_mel_features, noise_mel])

    """ dataset """
    train_set = TensorDataset(log_mel_features)
    dataset_sizes = {"train": len(train_set)}
    print("[INFO]train_set size: ", len(train_set))
    if len(train_set) == 0:
        sys.exit()

    """ dataloader """
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloaders = {"train": train_dataloader}

    """ set up model """
    model = MelVQVAE(codebook_size=codebook_size, downsample_ratio=downsample_ratio)

    model.to(device)

    """ training loop """

    ######################################################
    """ stage one training: mel autoencoding """
    ######################################################

    """ set up optimizer and scheduler """
    early_stopping = EarlyStopper(patience=4, min_delta=0.01)
    optimizer = build_optimizer(args, model, mode="finetune")
    T = args["T_max"]
    logger.info("T:{}".format(T))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)

    print("=== start autoencode ... ===")
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
