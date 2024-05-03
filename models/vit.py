import torch
from torch import nn, optim
import numpy as np
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
import lightning.pytorch as pl


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, dim, kernel_size=3, padding=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Reshape input tensor to fit the expected shape of Conv1d
        x = x.transpose(1, 2)  # Assuming x has shape (batch_size, dim, sequence_length)

        x = self.net(x)
        x = x.transpose(1, 2)  # Restore the original shape
        # print(x.shape)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            # print(x.shape)
            x = ff(x) + x
            # print(x.shape)
        return x


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class ViT(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        seq_len = configs["model_params"]["seq_len"]
        patch_size = configs["model_params"]["patch_size"]
        num_classes = configs["model_params"]["num_classes"]
        dim = configs["model_params"]["dim"]
        depth = configs["model_params"]["depth"]
        heads = configs["model_params"]["heads"]
        mlp_dim = configs["model_params"]["mlp_dim"]
        dropout = configs["model_params"]["dropout"]
        emb_dropout = configs["model_params"]["emb_dropout"]
        channels = configs["model_params"]["in_channels"]
        dim_head = configs["model_params"]["dim_head"]
        assert (seq_len % patch_size) == 0
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.val_step_outputs = []
        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)

    def _get_reconstruction_loss(self, batch):
        x, labels, _ = batch
        x_hat = self.forward(x)
        _loss = nn.SmoothL1Loss()
        loss = _loss(x_hat.flatten(), labels.flatten())
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=5, max_iters=200
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, train_batch, batch_idx):
        loss = self._get_reconstruction_loss(train_batch)
        self.training_step_outputs.append(loss)
        self.log("train_loss", loss)
        # print(loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._get_reconstruction_loss(val_batch)
        self.val_step_outputs.append(loss)
        self.log("val_loss", loss, "batch_size", batch_size=50)
        return loss


if __name__ == '__main__':
    v = ViT(
        seq_len=3520,
        patch_size=64,
        num_classes=1,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.2,
        emb_dropout=0.1
    )

    time_series = torch.randn(32, 1, 3520)
    logits = v(time_series)  # (4, 1000)
    print(logits.shape)
