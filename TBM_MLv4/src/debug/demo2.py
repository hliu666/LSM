debug_mode = False

# Switch to training mode from here
training_mode = True

import os
import sys
import numpy as np
import pandas as pd
import time
import random
import math
import pickle
from pickle import dump, load
import glob
import re
import string
import collections
import json
import gc

from sklearn.preprocessing import RobustScaler, StandardScaler

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Optimizer, Adam, lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from torchinfo import summary

import warnings

warnings.filterwarnings('ignore')

gc.enable()

rand_seed = 1120

print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch Lightning Version: {pl.__version__}")

num_workers = 4
gpus = [0]

breath_steps = 80

epochs = 1000 if not debug_mode else 1

train_batch_size = 128
infer_batch_size = 2048

mixed_precision = False

learning_rate = 1e-3
learning_rate *= len(gpus)

weight_decay = 0

# Model hyperparameters #
# Explanation reference: https://timeseriesai.github.io/tsai/models.TST.html

d_model = 128  # total dimension of the model (number of features created by the model) Usual values: 128-1024.
n_heads = 8  # parallel attention heads. Usual values: 8-16.
num_layers = 3  # the number of sub-encoder-layers in the encoder. Usual values: 2-8.
dim_feedforward = 256  # the dimension of the feedforward network model. Usual values: 256-4096.
dropout = 0.1  # amount of residual dropout applied in the encoder. Usual values: 0.-0.3.
pos_encoding = 'learnable'  # fixed, learnable
activation = 'gelu'  # # activation function of intermediate layer, relu or gelu.
norm = 'BatchNorm'  # BatchNorm, LayerNorm

lr_decay_steps = 983 * 300  # every K epochs
lr_decay_rate = 0.1  # every K epochs

mean_mask_length = 3  # Imputation: the desired mean length of masked segments. Used only when `mask_distribution` is 'geometric'.
masking_ratio = 0.15  # Imputation: mask this proportion of each variable
mask_mode = 'separate'  # Imputation: whether each variable should be masked separately
mask_distribution = 'geometric'  # Imputation: whether each mask sequence element is sampled independently at random

accumulate_grad_batches = 1
gradient_clip_val = 4.0

train_df = pd.read_csv(f"demo_data/train.csv")
test_df = pd.read_csv(f"demo_data/test.csv")
submit_df = pd.read_csv(f"demo_data/sample_submission.csv")
print(train_df.shape, test_df.shape, submit_df.shape)

target_column = "pressure"
meta_columns = ["breath_id", "time_step"]
raw_features = [
    c for c in train_df.columns
    if c not in ["id", target_column, "R", "C"] + meta_columns
]

# Reference: https://www.kaggle.com/hijest/gaps-features-tf-lstm-resnet-like-ff
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_pickle(obj, folder_path):
    dump(obj, open(folder_path, 'wb'), pickle.HIGHEST_PROTOCOL)


def load_pickle(folder_path):
    return load(open(folder_path, 'rb'))

# Reference: https://github.com/gzerveas/mvts_transformer/blob/master/src/optimizers.py
# From https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py
class RAdam(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params,
                      (list, tuple)) and len(params) > 0 and isinstance(
                          params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0]
                                         or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2**state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 -
                                                                       beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) *
                            (N_sma - 2) / N_sma * N_sma_max /
                            (N_sma_max - 2)) / (1 - beta1**state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1**state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'],
                                         p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg,
                                         denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'],
                                         p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss

def add_features(df):
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df = pd.get_dummies(df)
    return df

train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)

train_features = add_features(train_df[['time_step'] + raw_features +
                                       ['R', 'C']].copy())
test_features = add_features(test_df[['time_step'] + raw_features +
                                     ['R', 'C']].copy())

train_indices = train_df.index.to_numpy().reshape(-1, breath_steps)
oof_df = train_df[["id", "pressure"]].copy()

scaler = StandardScaler(with_mean=True, with_std=True)
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# (batch, steps, features)
train_features = train_features.reshape(-1, breath_steps,
                                        train_features.shape[-1])
test_features = test_features.reshape(-1, breath_steps,
                                      test_features.shape[-1])

train_u_out = train_df[['u_out']].to_numpy().reshape(-1, breath_steps)
test_u_out = test_df[['u_out']].to_numpy().reshape(-1, breath_steps)
targets = train_df[['pressure']].to_numpy().reshape(-1, breath_steps)

train_breath_ids = train_df["breath_id"].unique()
print(train_features.shape, test_features.shape, train_u_out.shape, test_u_out.shape, targets.shape)

if training_mode:
    all_features = np.concatenate([train_features, test_features], axis=0)
    all_u_out = np.concatenate([train_u_out, test_u_out], axis=0)
    print(all_features.shape, all_u_out.shape)

del train_df, test_df
gc.collect()

# Reference: https://github.com/gzerveas/mvts_transformer

def noise_mask(X,
               masking_ratio,
               lm=3,
               mode='separate',
               distribution='geometric',
               exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(
                        X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(
                np.expand_dims(
                    geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1),
                X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]),
                                    size=X.shape,
                                    replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(
                np.random.choice(np.array([True, False]),
                                 size=(X.shape[0], 1),
                                 replace=True,
                                 p=(1 - masking_ratio, masking_ratio)),
                X.shape[1])

    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
        1 - masking_ratio
    )  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() >
                masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[
            i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val(
    )  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len,
                         device=lengths.device).type_as(lengths).repeat(
                             batch_size, 1).lt(lengths.unsqueeze(1)))

# Modified from: https://github.com/gzerveas/mvts_transformer/blob/master/src/datasets/dataset.py

def collate_unsuperv(data, max_len=None, mask_compensation=False):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, mask).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 ignore (padding)
    """

    batch_size = len(data)
    features, masks, u_out = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features
               ]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(
        batch_size, max_len,
        features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    target_masks = torch.zeros_like(
        X, dtype=torch.bool
    )  # (batch_size, padded_length, feat_dim) masks related to objective
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    targets = X.clone()
    X = X * target_masks  # mask input
    if mask_compensation:
        X = compensate_masking(X, target_masks)

    padding_masks = torch.zeros(
        batch_size, max_len, dtype=torch.bool)  # (batch_size, padded_length)
    for i in range(batch_size):
        padding_masks[i, :] = torch.where(u_out[i] == 0, 1, 0)

    target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict

    return X, targets, target_masks, padding_masks


class VPPMaskedInputDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features,
        u_out,
        mean_mask_length=3,
        masking_ratio=0.15,
        mode='separate',
        distribution='geometric',
    ):
        self.features = features
        self.u_out = u_out

        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.mode = mode
        self.distribution = distribution

    def __getitem__(self, index):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            index: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            mask: (seq_length, feat_dim) boolean tensor: 0s mask and predict, 1s: unaffected input
        """

        X = self.features[index, :, :]  # (seq_length, feat_dim) array

        mask = noise_mask(X, self.masking_ratio, self.mean_mask_length,
                          self.mode, self.distribution,
                          None)  # (seq_length, feat_dim) boolean array

        return torch.from_numpy(X), torch.from_numpy(mask), torch.from_numpy(
            self.u_out[index, :])

    def update(self):
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return self.features.shape[0]

class VPPTestDataset(torch.utils.data.Dataset):
    def __init__(self, data, u_out):
        self.X = data
        self.u_out = u_out

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index, :, :], self.u_out[index, :]


class MaskedMAELoss(nn.Module):
    """ Masked MAE Loss
    """
    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mae_loss = nn.L1Loss(reduction=self.reduction)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mae_loss(masked_pred, masked_true)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """
    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)

from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError(
        "activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer(
            'pe', pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(
            max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError(
        "pos_encoding should be 'learnable'/'fixed', not '{}'".format(
            pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(
            d_model, eps=1e-5
        )  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src,
                              src,
                              src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):
    def __init__(self,
                 feat_dim,
                 max_len,
                 d_model,
                 n_heads,
                 num_layers,
                 dim_feedforward,
                 dropout=0.1,
                 pos_encoding='fixed',
                 activation='gelu',
                 norm='BatchNorm',
                 freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model,
                                                     dropout=dropout *
                                                     (1.0 - freeze),
                                                     max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model,
                                                    self.n_heads,
                                                    dim_feedforward,
                                                    dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model
        )  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(
            inp, src_key_padding_mask=~padding_masks
        )  # (seq_length, batch_size, d_model)
        output = self.act(
            output
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(
            output)  # (batch_size, seq_length, feat_dim)

        return output

class TSTLightning(pl.LightningModule):
    def __init__(
            self,
            fold=None,
            training_set=None,
            in_features=None,
            out_features=breath_steps,
            d_model=128,  # total dimension of the model (number of features created by the model) Usual values: 128-1024.
            n_heads=8,  # parallel attention heads. Usual values: 8-16.
            num_layers=3,  # the number of sub-encoder-layers in the encoder. Usual values: 2-8.
            dim_feedforward=256,  # the dimension of the feedforward network model. Usual values: 256-4096.
            dropout=0.1,  # amount of residual dropout applied in the encoder. Usual values: 0.-0.3.
            pos_encoding='fixed',  # fixed, learnable
            activation='gelu',  # # activation function of intermediate layer, relu or gelu.
            norm='BatchNorm',  # BatchNorm, LayerNorm
            learning_rate=1e-3):
        super(TSTLightning, self).__init__()

        self.fold = fold
        self.training_set = training_set

        self.learning_rate = learning_rate

        self.model = TSTransformerEncoder(
            feat_dim=in_features,
            max_len=breath_steps,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,  # fixed, learnable
            activation=activation,  # relu, gelu
            norm=norm,  # BatchNorm, LayerNorm
            freeze=False)

        self.num_parameters = count_parameters(self.model)
        print(f"Trainable params: {self.num_parameters:,}")

        self.loss_fn = MaskedMSELoss(reduction="mean")

        # Save passed hyperparameters
        self.save_hyperparameters("in_features", "d_model", "n_heads",
                                  "num_layers", "dim_feedforward", "dropout",
                                  "pos_encoding", "activation", "norm",
                                  "learning_rate")

        # Important: Activates manual optimization
        # https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html#manual-optimization
        self.automatic_optimization = False

    def forward(self, x, masks):
        # print(x.shape, masks.shape)
        return self.model(x, masks)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        opt.zero_grad(set_to_none=True)

        X, targets, target_masks, padding_masks = batch

        logits = self(X, padding_masks)  # (batch_size, breath_steps)

        # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
        target_masks = target_masks * padding_masks.unsqueeze(-1)

        loss = self.loss_fn(
            logits, targets, target_masks
        )  # (num_active,) individual loss (square error per element) for each active value in batch

        current_lr = self.lr_schedulers().get_last_lr()[0]

        self.manual_backward(loss)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(),
                                                   gradient_clip_val)

        opt.step()

        scheduler = self.lr_schedulers()
        scheduler.step()

        self.log_dict({
            'train_loss': loss,
            'learning_rate': current_lr
        },
                      on_step=True,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, val_step_outputs):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # Extract z representation features
        with torch.no_grad():
            X, u_out = batch

            batch_size = X.shape[0]
            padding_masks = torch.zeros(
                batch_size, breath_steps, dtype=torch.bool,
                device=self.device)  # (batch_size, padded_length)
            for i in range(batch_size):
                padding_masks[i, :] = torch.where(u_out[i] == 0, 1, 0)

            logits = self(X.float(),
                          padding_masks)  # (batch_size, breath_steps)
            return logits.detach().cpu()

    def setup(self, stage=None):
        if self.training:
            self.train_dataset = VPPMaskedInputDataset(self.training_set[0],
                                                       self.training_set[1])

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda x: collate_unsuperv(x, max_len=breath_steps),
            drop_last=False)
        print(f"Train iterations: {len(train_dataloader)}")
        return train_dataloader

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def configure_optimizers(self):
        print(f"Initial Learning Rate: {self.hparams.learning_rate:.6f}")

        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_epsilon = 1e-8
        optimizer = RAdam(self.parameters(),
                          lr=self.hparams.learning_rate,
                          betas=(adam_beta1, adam_beta2),
                          eps=adam_epsilon,
                          weight_decay=weight_decay,
                          degenerated_to_sgd=True)

        train_steps = epochs * (len(self.train_dataloader()) //
                                accumulate_grad_batches)
        print(f"Total number of training steps: {train_steps}")

        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(lr_decay_steps, train_steps,
                                  lr_decay_steps)),
            gamma=lr_decay_rate)

        return [optimizer], [scheduler]

def get_model(fold_i,
              training_set,
              in_features=None,
              model_path=None,
              print_model=False):

    if training_mode:
        model = TSTLightning(
            fold=fold_i,
            training_set=training_set,
            in_features=in_features,
            out_features=breath_steps,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,  # fixed, learnable
            activation=activation,  # relu, gelu
            norm=norm,  # BatchNorm, LayerNorm
            learning_rate=learning_rate)
        if print_model:
            print(model)
    else:
        model = TSTLightning.load_from_checkpoint(
            model_path,
            fold=fold_i,
            training_set=training_set,
            in_features=in_features,
        )

        model.freeze()
        model.eval()
    return model

# Ensure Reproducibility
seed_everything(rand_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if training_mode:
    print(f"Unsupervised Train Shape: {all_features.shape}, {all_u_out.shape}")
    model_output_folder = 'demo_data'
    logger = TensorBoardLogger(model_output_folder,
                               name=f"logs",
                               default_hp_metric=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{model_output_folder}",
        filename="{epoch}-{train_loss_epoch:.6f}",
        save_top_k=1,
        save_weights_only=True,
        save_last=False,
        verbose=True,
        monitor='train_loss_epoch',
        mode='min')

    callbacks = [checkpoint_callback]

    model = get_model(fold_i=None,
                      training_set=(all_features, all_u_out),
                      in_features=all_features.shape[-1])

    trainer = Trainer(
        gpus=gpus if torch.cuda.is_available() else None,
        distributed_backend="dp"
        if torch.cuda.is_available() else None,  # multiple-gpus, 1 machine
        max_epochs=epochs,
        benchmark=False,
        deterministic=True,
        log_gpu_memory=False,
        checkpoint_callback=True,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=16 if mixed_precision and torch.cuda.is_available() else 32,
        logger=logger)

    trainer.fit(model)

    del model, trainer

    torch.cuda.empty_cache()
    gc.collect()



