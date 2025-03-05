import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from typing import Mapping, Text, Tuple
from torch.cuda.amp import autocast
from einops.einops import rearrange
from einops.layers.torch import Rearrange
from argparse import Namespace
from utils.utils import ssim

from loss.ssim import MS_SSIM_L1_mix_loss
from loss.clip import CLIPLoss
from modules.vae import Encoder, Decoder
from peft import AdaLoraModel


## 定义一个装饰器，查看输入数据是否存在nan或者inf
def check_nan_inf(input_data, text):
    # 检查是否存在NaN
    nan_check = torch.isnan(input_data)
    if nan_check.any().item():
        print(text, "中输入数据中存在NaN")

    # 检查是否存在Inf
    inf_check = torch.isinf(input_data)
    if inf_check.any().item():
        print(text, "中输入数据中存在Inf")


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, cls_token=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (5000,840)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (5000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (420,) # 840/2
        pe[:, 0::2] = torch.sin(position * div_term)  #
        pe[:, 1::2] = torch.cos(position * div_term)
        if cls_token:
            pe = torch.cat([torch.zeros((1, d_model)), pe], 0)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (5000,1,840)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print('[DEBUG] input size:', x.size())  # {Tensor:(16,57,840)}
        # print('[DEBUG] positional embedding size:', self.pe.size())  # {Tensor:(5000,1,840)}
        x = x + self.pe[: x.size(0), :]
        # print('[DEBUG] output x with pe size:', x.size())
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, cls_token=False):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        scale = d_model**-0.5
        self.pe = nn.Parameter(scale * torch.randn(max_len, d_model))
        if cls_token:
            class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
            self.pe = torch.cat([class_embedding, self.pe], 0)

    def forward(self, x):
        # print('[DEBUG] input size:', x.size())  # {Tensor:(16,57,840)}
        # print('[DEBUG] positional embedding size:', self.pe.size())  # {Tensor:(5000,1,840)}
        x = x + self.pe[: x.size(1), :]
        # print('[DEBUG] output x with pe size:', x.size())
        return self.dropout(x)


class NeuSpeech(nn.Module):
    def __init__(self, pretrained_layers):
        super(NeuSpeech, self).__init__()
        self.pretrained = pretrained_layers

    def forward(self, rawEEG_batch, target_ids_batch_converted, decoder_input_ids_batch=None):
        b, c, t = rawEEG_batch.shape
        if t < 6000:
            pad = torch.zeros((b, c, 6000 - t)).to(rawEEG_batch.device)
            rawEEG_batch = torch.cat([rawEEG_batch, pad], -1)
            del pad

        out = self.pretrained(
            input_features=rawEEG_batch,
            return_dict=True,
            labels=target_ids_batch_converted[:, 1:],
            decoder_input_ids=decoder_input_ids_batch,
        )
        return out

    @torch.no_grad()
    def generate(
        self,
        rawEEG_batch,
        target_ids_batch_converted,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        b, c, t = rawEEG_batch.shape
        if t < 6000:
            pad = torch.zeros((b, c, 6000 - t)).to(rawEEG_batch.device)
            rawEEG_batch = torch.cat([rawEEG_batch, pad], -1)
            del pad

        output = self.pretrained.generate(
            input_features=rawEEG_batch,
            labels=target_ids_batch_converted[:, 1:],
            return_dict=True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )

        return Namespace(predictions=output)


class ResBlock(nn.Module):
    # init()：进行初始化，申明模型中各层的定义
    # downsample=None对应实线残差结构，否则为虚线残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # 使用批量归一化
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 使用ReLU作为激活函数
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        if downsample is None and in_channel != out_channel:
            self.downsample = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, stride), nn.BatchNorm2d(out_channel))

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 残差块保留原始输入
        identity = x
        # 如果是虚线残差结构，则进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # -----------------------------------------
        out = self.conv2(out)
        out = self.bn2(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class VectorQuantizer(torch.nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        token_size: int = 256,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = False,
    ):
        super().__init__()
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm

    # Ensure quantization is performed using f32
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = rearrange(z, "b h w c -> (b h w) c")

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, embedding.T)
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) ** 2)
        codebook_loss = torch.mean((z_quantized - z.detach()) ** 2)

        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, "b h w c -> b c h w").contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices.view(
                z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3]
            ),
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum("bd,dn->bn", indices, self.embedding.weight)
        else:
            raise NotImplementedError
        return z_quantized


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        """
        n_embeddings: 码本大小，即码字总数
        embedding_dim: 每个码字的维度
        commitment_cost: commitment loss前的系数，即commitment cost
        decay: EMA更新公式中的\gamma
        epsilon: 防止除数为0
        """
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        ### 初始化码本 ###
        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        # 从均匀分布U[-1/n_embeddings,1/n_embeddings]中抽样数值对tensor进行填充
        embedding.uniform_(-init_bound, init_bound)

        ### 设置一些参数 ###
        # self.register_buffer('name', Tensor)定义一组名为name的参数，该组参数的特别之处在于：
        # 调用optimizer.step()后该组参数不会变化，只可人为地改变它们的值；
        # 但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w->b (h w) c")
        K, D = self.embedding.size()  # K表示码字总数/码本大小，D表示码字维度
        x_flat = x.detach().reshape(-1, D)  # x:[B,T,D]->x_flat:[BxT,D]

        # torch.addmm(M,M1,M2,a,b) = bM+a(M1@M2), 其中M1@M2表示矩阵乘法
        # 计算序列x和码本中各码字之间的距离
        distances = torch.addmm(
            torch.sum(self.embedding**2, dim=1) + torch.sum(x_flat**2, dim=1, keepdim=True),
            x_flat,
            self.embedding.t(),
            alpha=-2.0,
            beta=1.0,
        )

        # 选择距离最近的码字，获得的indices为相应码字的索引序列
        indices = torch.argmin(distances.float(), dim=-1)

        # F.one_hot(indices, K)是对indices进行one-hot编码
        # 例，F.one_hot([5,3,2,4,1], 6)将[5,3,2,4,1]编码为
        # [[0,0,0,0,0,1]
        #  [0,0,0,1,0,0]
        #  [0,0,1,0,0,0]
        #  [0,0,0,0,1,0]
        #  [0,1,0,0,0,0]]
        encodings = F.one_hot(indices, K).float()  # encodings为索引序列indices的one-hot编码

        ### 获得相应的码字 ###
        # F.embedding(indices, self.embedding)用于使用索引indices在固定码本self.embedding中检索码字
        quantized = F.embedding(indices, self.embedding)  # quantized为检索到的相应的码字
        quantized = quantized.view_as(x)  # [BxT,D]->[B,T,D]

        ### 使用EMA方法更新码本 ###
        if self.training:
            # self.ema_count即为EMA更新公式中的N(t)，其中的第i个元素表示所有数据中与第i个码字对应的连续变量x_i的数量
            # torch.sum(encodings, dim=0)即EMA更新公式中的n(t)，其中的第i个元素表示当前batch中与第i个码字对应的连续变量x_i的数量
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + D * self.epsilon) * n
            # dw即EMA更新公式中的\sum{z_{i,j}}，第i个元素即当前batch中与第i个码字对应的连续元素的和
            dw = torch.matmul(encodings.t(), x_flat)
            # self.ema_weight即EMA更新公式中的m(t)，第i个元素即所有batch中与第i个码字对应的连续变量的和
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            # 更新码本中的码字
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        ### 计算VQ Loss ###
        # VQ损失，固定quantized(因为这一项已通过上述EMA方法更新)，使x向quantized更靠近
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        ### 使用stop-gradient operator, 便于反向传播计算梯度###
        # .detach()即使用了stop-gradient operator，在反向传播的时候只计算对x的梯度
        quantized = x + (quantized - x).detach()

        quantized = rearrange(quantized, "b (h w) c->b c h w", h=h)

        result_dict = dict(quantizer_loss=loss, min_encoding_indices=indices.view(b, h, w))

        return quantized, result_dict


class MelVQVAE(nn.Module):
    def __init__(self, num_res_blocks=0, codebook_size=2048, latent_channels=8, downsample_ratio=4):
        super(MelVQVAE, self).__init__()

        ch_mult = (1,) + (2,) * (int(math.log2(downsample_ratio)) - 1) + (4,)

        self.quantizer = VectorQuantizer(codebook_size, latent_channels, use_l2_norm=True)
        # self.quantizer = VQEmbeddingEMA(codebook_size, latent_channels)
        self.codebook_size = codebook_size

        self.mel_encoder = Encoder(ch=32, ch_mult=ch_mult, num_res_blocks=1, double_z=False)
        self.mel_decoder = Decoder(ch=32, ch_mult=ch_mult, num_res_blocks=num_res_blocks, z_channels=latent_channels)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights.

        Args:
            module (torch.nn.Module): module to initialize
        """        
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, mel):
        mel = mel.unsqueeze(1)

        return self.mel_encoder(mel)

    def quantize(self, z):
        z_quantized, result_dict = self.quantizer(z)
        return z_quantized, result_dict

    def decode(self, z):
        x = self.mel_decoder(z)
        x = rearrange(x, "b 1 d t->b d t")

        return x

    def forward(
        self,
        target_mel_feature_batch,
        loss="mse",
    ):
        z = self.encode(target_mel_feature_batch)
        z_quantized, result_dict = self.quantize(z)
        total_loss = 0.5 * result_dict["quantizer_loss"]
        indices = result_dict["min_encoding_indices"]
        recon_mel = self.decode(z_quantized)

        if loss == "mse":
            criterion = nn.MSELoss(reduction="none")
            mel_loss = criterion(recon_mel, target_mel_feature_batch)
            mel_loss = mel_loss.mean()
            total_loss += mel_loss
        elif loss == "ssim":
            sim = ssim(recon_mel.unsqueeze(1), target_mel_feature_batch.unsqueeze(1))
            mel_loss = 1 - sim
            total_loss += mel_loss
        elif loss == "ms_ssim_l1_mix":
            criterion = MS_SSIM_L1_mix_loss(win_size=5)
            encoded_embedding_img = recon_mel.unsqueeze(1).repeat(1, 3, 1, 1)
            target_mel_feature_batch = target_mel_feature_batch.unsqueeze(1).repeat(1, 3, 1, 1)
            mel_loss = criterion(encoded_embedding_img, target_mel_feature_batch).mean()
            total_loss += mel_loss
        else:
            raise Exception("loss not found")
        return Namespace(loss=total_loss, mel_loss=mel_loss, recon_mel=recon_mel, indices=indices)


class BrainECHO(nn.Module):
    def __init__(
        self,
        pretrained_layers,
        in_chan=64,
        d_model=256,
        n_head=8,
        ffn_dim=2048,
        num_layers=4,
        mel_len=1200,
        codebook_size=2048,
        latent_channels=8,
        r=4,
        use_mlp=False,
        mel_interp=False,
        split_mel=False,
    ):
        super(BrainECHO, self).__init__()

        self.task = None
        self.pretrained = pretrained_layers
        self.mel_len = mel_len
        self.mel_interp = mel_interp
        self.split_mel = split_mel

        self.TS_Conv = nn.Sequential(
            nn.Conv2d(1, 64, (1, 5), (1, 2), (0, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, d_model, (in_chan, 1), 1),
            nn.BatchNorm2d(d_model),
            nn.ELU(inplace=True),
        )
        for i in range((int(math.log2(r)) - 2)):
            self.TS_Conv.add_module(
                f"conv_block_{i}",
                nn.Sequential(
                    nn.Conv2d(d_model, d_model, (1, 3), (1, 2), (0, 1)),
                    nn.BatchNorm2d(d_model),
                    nn.ELU(inplace=True),
                ),
            )
        self.TS_Conv.add_module("rearrange", Rearrange("b c h w->b (h w) c"))

        eeg_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(eeg_encoder_layer, num_layers=num_layers)
        self.pos_embed_e = LearnablePositionalEncoding(d_model, max_len=mel_len // r * 2)
        self.mel_vqvae = MelVQVAE(codebook_size=codebook_size, downsample_ratio=r)
        for p in self.mel_vqvae.parameters():
            p.requires_grad = False
        self.mel_vqvae.eval()

        if r > 2:
            self.conv_out = nn.Conv2d(1, latent_channels, (1, 3), (1, 2), (0, 1))
        else:
            self.conv_out = nn.Conv2d(1, latent_channels, (1, 3), (1, 1), (0, 1))
        self.r = r

        self.fc_eeg = nn.Linear(d_model, 80 // r)
        if not use_mlp:
            self.mid_fc = ResidualAdd(
                nn.Sequential(nn.Linear(80 // r * mel_len // r, 80 // r * mel_len // r), nn.GELU())
            )
        else:
            self.mid_fc = ResidualAdd(
                nn.Sequential(
                    nn.Linear(80 // r * mel_len // r, ffn_dim),
                    nn.LayerNorm(ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, 80 // r * mel_len // r),
                )
            )
        self.ln = nn.LayerNorm(80 // r * mel_len // r)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights.

        Args:
            module (torch.nn.Module): module to initialize
        """        
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def add_adalora_pretrained_model(self, pretrained, config):
        pretrained.model.encoder = AdaLoraModel(pretrained.model.encoder, config, "default")
        self.pretrained = pretrained

    def encode(self, rawEEG):
        rawEEG = rawEEG.unsqueeze(1)
        eeg_embedding = self.TS_Conv(rawEEG)
        eeg_embedding = self.pos_embed_e(eeg_embedding)
        eeg_embedding = self.transformer(eeg_embedding)
        eeg_embedding = F.gelu(self.fc_eeg(eeg_embedding))
        eeg_embedding = rearrange(eeg_embedding, "b l d->b 1 d l")
        eeg_embedding = F.elu(self.conv_out(eeg_embedding), inplace=True)
        eeg_embedding = rearrange(eeg_embedding, "b c h w->b c (h w)")
        eeg_embedding = self.mid_fc(eeg_embedding)
        eeg_embedding = self.ln(eeg_embedding)
        eeg_embedding = rearrange(eeg_embedding, "b c (h w)->b c h w", h=80 // self.r)

        return eeg_embedding

    def decode(self, z):
        x = self.mel_vqvae.decode(z)

        return x

    def forward(
        self,
        rawEEG_batch,
        target_ids_batch_converted,
        target_mel_feature_batch,
        decoder_input_ids_batch=None,
        loss="mse",
    ):
        """_summary_

        Args:
            rawEEG_batch (Tensor): batch_size * channels * time
            target_ids_batch_converted (Tensor): batch_size * max_length
            target_mel_feature_batch (Tensor): batch_size * 80 * time_frames
            decoder_input_ids_batch (Tensor, optional): batch_size * max_length. Defaults to None.
            loss (str, optional): latent alignment criterion. Defaults to "mse".

        Returns:
            Namespace: loss, reconstructed mel spectrograms or codebook indices
        """        

        z = self.encode(rawEEG_batch)
        z_quantized, result_dict = self.mel_vqvae.quantize(z)
        total_loss = 0.5 * result_dict["quantizer_loss"]
        indices = result_dict["min_encoding_indices"]
        recon_mel = self.decode(z_quantized)

        if self.task == "task1":
            with torch.no_grad():
                mel_z = self.mel_vqvae.encode(target_mel_feature_batch)
            if loss == "mse":
                latent_loss = F.mse_loss(z, mel_z)
            elif loss == "cos_sim":
                z = rearrange(z, "b c h w->(b h w) c")
                mel_z = rearrange(mel_z, "b c h w->(b h w) c")
                target = torch.ones(z.shape[0]).to(z.device)
                latent_loss = F.cosine_embedding_loss(z, mel_z, target)
            elif loss == "clip_loss":
                criterion = CLIPLoss()
                z = rearrange(z, "b c h w->(b h) w c")
                mel_z = rearrange(mel_z, "b c h w->(b h) w c")
                latent_loss = criterion(z, mel_z)
            else:
                raise Exception("loss not found")

            del mel_z
            mel_loss = F.mse_loss(recon_mel, target_mel_feature_batch)
            total_loss = total_loss + latent_loss + mel_loss

            return Namespace(loss=total_loss, mel_loss=mel_loss, recon_mel=recon_mel, indices=indices)

        elif self.task == "task2":
            if self.mel_interp:
                recon_mel = F.interpolate(recon_mel.unsqueeze(1), (80, self.mel_len * 2), mode="bilinear").squeeze(1)
            if self.split_mel:
                recon_mel = rearrange(recon_mel, "(b p) d t->b d (p t)", p=1200 // self.mel_len)
            
            b, d, t = recon_mel.shape
            # pad spectrogram to 3000 time slots
            pad = -1 * torch.ones((b, d, 3000 - t)).to(recon_mel.device)
            recon_mel = torch.cat([recon_mel, pad], -1)

            out = self.pretrained(
                input_features=recon_mel,
                return_dict=True,
                labels=target_ids_batch_converted[:, 1:],
                decoder_input_ids=decoder_input_ids_batch,
            )
            return Namespace(loss=out.loss, logits=out.logits, indices=indices)

    @torch.no_grad()
    def generate(
        self,
        rawEEG_batch,
        target_ids_batch_converted,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        z = self.encode(rawEEG_batch)
        z_quantized, result_dict = self.mel_vqvae.quantize(z)
        indices = result_dict["min_encoding_indices"]
        recon_mel = self.decode(z_quantized)

        if self.mel_interp:
            recon_mel = F.interpolate(recon_mel.unsqueeze(1), (80, self.mel_len * 2), mode="bilinear").squeeze(1)
        if self.split_mel:
            recon_mel = rearrange(recon_mel, "(b p) d t->b d (p t)", p=1200 // self.mel_len)
        b, d, t = recon_mel.shape
        pad = -1 * torch.ones((b, d, 3000 - t)).to(recon_mel.device)
        recon_mel = torch.cat([recon_mel, pad], -1)

        output = self.pretrained.generate(
            input_features=recon_mel,
            labels=target_ids_batch_converted[:, 1:],
            return_dict=True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )

        return Namespace(predictions=output, recon_mel=recon_mel, indices=indices)
