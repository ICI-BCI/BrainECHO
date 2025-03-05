import random
import torch
import json
import logging
import os
import pickle
import datetime
import numpy as np
import wandb

import yaml
import re
import sys
from torch import optim
from transformers.optimization import Adafactor, AdafactorSchedule
from logging import getLogger

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

BOS_ID, PAD_ID, EOS_ID, MASK_ID = 0, 1, 2, 50264


def pretty(d):
    return json.dumps(d, indent=4, ensure_ascii=False)


def wandb_init(config, project):
    wandb.init(
        project=project,
        config=dict(config)
    )
    wandb.run.name = config["exp_name"]


def convert_config_dict(config_dict):
    """This function convert the str parameters to their original type.
    """
    for key in config_dict:
        param = config_dict[key]
        if not isinstance(param, str):
            continue
        try:
            value = eval(param)
            if not isinstance(value, (str, int, float, list, tuple, dict, bool)):
                value = param
        except (NameError, SyntaxError, TypeError):
            if isinstance(param, str):
                if param.lower() == "true":
                    value = True
                elif param.lower() == "false":
                    value = False
                else:
                    value = param
            else:
                value = param
        config_dict[key] = value
    return config_dict


def read_configuration(config_file):
    # read configuration from yaml file
    yaml_loader = yaml.FullLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        yaml_config_dict = yaml.load(f.read(), Loader=yaml_loader)

    # read configuration from cmd line
    cmd_config_dict = dict()
    unrecognized_args = []
    if "ipykernel_launcher" not in sys.argv[0]:
        for arg in sys.argv[1:]:
            if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                unrecognized_args.append(arg)
                continue
            cmd_arg_name, cmd_arg_value = arg[2:].split("=")
            if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
            else:
                cmd_config_dict[cmd_arg_name] = cmd_arg_value
    if len(unrecognized_args) > 0:
        logger = getLogger()
        # logger.warning('command line args [{}] will not be used in TextBox'.format(' '.join(unrecognized_args)))

    cmd_config_dict = convert_config_dict(cmd_config_dict)

    final_config_dict = dict()
    final_config_dict.update(yaml_config_dict)
    final_config_dict.update(cmd_config_dict)

    return final_config_dict


def build_optimizer(config, model, mode):
    if mode == "finetune":
        lr = config["lr_finetune"]
        n_optimizer = config["finetune_optimizer"].lower()
    elif mode == "step_clip":
        try:
            lr = config["step_clip_lr"]
            n_optimizer = config["step_clip_optimizer"].lower()
        except:
            lr = config["lr_clip"]
            n_optimizer = config["step_clip_optimizer"].lower()
    elif mode == "cet-mae":
        lr = config["cet_mae_lr"]
        n_optimizer = config["cet_mae_optimizer"].lower()
    elif mode == "cscl":
        lr = config["step1_lr"]
        n_optimizer = config["step1_optimizer"].lower()
    else:
        lr = config["lr"]
        n_optimizer = config["optimizer"].lower()

    print(f"Mode: {mode}, Optimizer: {n_optimizer} Learning rate: {lr}")

    parameters = [p for p in model.parameters() if p.requires_grad]
    if n_optimizer == 'adam':
        optimizer = optim.Adam(parameters, lr=lr)
    elif n_optimizer == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr)
    elif n_optimizer == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=lr)
    elif n_optimizer == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=lr)
    elif n_optimizer == 'adamw':
        # gen_params = []
        # classify_params = []
        # for name, p in model.named_parameters():
        #     if 'linear_proc' in name:
        #         classify_params.append(p)
        #     else:
        #         gen_params.append(p)
        # pretrained_params = []
        # other_params = []
        # for name, p in model.named_parameters():
        #     if 'pretrained' in name:
        #         pretrained_params.append(p)
        #     else:
        #         other_params.append(p)
        # decoder_params = []
        # other_params = []
        # for name, p in model.named_parameters():
        #     if p.requires_grad:
        #         if 'decoder_blocks' in name or 'decoder_norm' in name or 'decoder_pred' in name:
        #             decoder_params.append(p)
        #         else:
        #             other_params.append(p)
        optimizer = optim.AdamW(parameters, lr=lr)
        # optimizer = optim.AdamW([{'params': gen_params, 'lr': lr}, {'params': classify_params, 'lr': 2e-4}])
        # optimizer = optim.AdamW([{'params': other_params, 'lr': lr}, {'params': pretrained_params, 'lr': 2e-7}])
        # optimizer = optim.AdamW([{'params': other_params, 'lr': lr}, {'params': decoder_params, 'lr': 2e-7}])
    elif n_optimizer == 'adafactor':
        optimizer = Adafactor(parameters, scale_parameter=False, relative_step=False, warmup_init=False, lr=lr)
    else:
        raise ValueError('Received unrecognized optimizer {}.'.format(config["optimizer"].lower()))
    return optimizer


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def init_logger(config):
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])

    logfilename = '{}-{}.log'.format(config["model_name"], get_local_time())
    logfilepath = os.path.join(config["log_dir"], logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config["state"] is None or config["state"].lower() == 'info':
        level = logging.INFO
    elif config["state"].lower() == 'debug':
        level = logging.DEBUG
    elif config["state"].lower() == 'error':
        level = logging.ERROR
    elif config["state"].lower() == 'warning':
        level = logging.WARNING
    elif config["state"].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        handlers=[fh, sh]
    )


def init_seed(seed, reproducibility):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def init_device(config):
    use_gpu = config["use_gpu"]
    device = torch.device("cuda:" + str(config["gpu_id"]) if torch.cuda.is_available() and use_gpu else "cpu")
    return device


def maybe_create(saved_dir):
    """check checkpoint path"""
    if not os.path.exists(os.path.join(saved_dir, "best")):
        os.makedirs(os.path.join(saved_dir, "best"))

    if not os.path.exists(os.path.join(saved_dir, "last")):
        os.makedirs(os.path.join(saved_dir, "last"))


def get_receptive_field(kernels, strides):
    assert len(kernels) == len(strides)
    l = 1
    s = 1
    for i in range(len(kernels)):
        l += (kernels[i] - 1) * s
        s *= strides[i]
    return l


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=np.float32)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def topk_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).int().sum(0, keepdim=True)
            res.append(correct_k)
        return torch.cat(res)


def remove_consecutive_duplicates(sentence):
    words = sentence.split()
    new_words = []

    if words:
        new_words.append(words[0])
    for word in words[1:]:
        if word != new_words[-1]:
            new_words.append(word)

    new_sentence = ' '.join(new_words)

    return new_sentence


def add_gaussian_noise(signal_input, snr_range):
    # 获取信号的形状和通道数
    ch, length = signal_input.shape

    # 生成每个通道的信噪比
    snr_per_channel = np.random.uniform(*snr_range, size=ch)

    # 初始化噪声信号
    noise_signal = np.zeros_like(signal_input)

    # 逐通道添加高斯噪声
    for i in range(ch):
        # 计算当前通道的信噪比
        snr = snr_per_channel[i]

        # 计算当前通道的噪声标准差
        noise_std = np.sqrt(np.mean(signal_input[i] ** 2) / (10 ** (snr / 10)))

        # 生成高斯噪声
        noise = np.random.normal(scale=noise_std, size=length)

        # 添加噪声到当前通道
        noise_signal[i] = signal_input[i] + noise

    # 应用噪声
    noisy_signal = signal_input + noise_signal

    return noisy_signal