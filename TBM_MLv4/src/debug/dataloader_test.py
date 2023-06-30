import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader

# Reference: https://www.kaggle.com/hijest/gaps-features-tf-lstm-resnet-like-ff
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
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
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def compensate_masking(X, mask):
    """
    Compensate feature vectors after masking values, in a way that the matrix product W @ X would not be affected on average.
    If p is the proportion of unmasked (active) elements, X' = X / p = X * feat_dim/num_active
    Args:
        X: (batch_size, seq_length, feat_dim) torch tensor
        mask: (batch_size, seq_length, feat_dim) torch tensor: 0s means mask and predict, 1s: unaffected (active) input
    Returns:
        (batch_size, seq_length, feat_dim) compensated features
    """

    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active


def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
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
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

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
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


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
    features, masks, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    target_masks = torch.zeros_like(X,
                                    dtype=torch.bool)  # (batch_size, padded_length, feat_dim) masks related to objective
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    targets = X.clone()
    X = X * target_masks  # mask input
    if mask_compensation:
        X = compensate_masking(X, target_masks)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
    return X, targets, target_masks, padding_masks, IDs


class ClassiregressionDataset(Dataset):

    def __init__(self, data, indices):
        super(ClassiregressionDataset, self).__init__()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data.feature_df.loc[self.IDs]

        self.labels_df = self.data.labels_df.loc[self.IDs]

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (num_labels,) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """

        X = self.feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        y = self.labels_df.loc[self.IDs[ind]].values  # (num_labels,) array

        return torch.from_numpy(X), torch.from_numpy(y), self.IDs[ind]

    def __len__(self):
        return len(self.IDs)

class Dataloader:
    def __init__(self, dC, v, p):

        carp_train_features, carp_train_out, carp_test_features, carp_test_out, self.carp_scaler = self.dataLoader(
            dC.daily_df,
            p.lookback_daily,
            p.test_portion,
            v.y_carp_vars,
            v.x_carp_vars,
            v.x_carp_pars)

        rtmo_train_features, rtmo_train_out, rtmo_test_features, rtmo_test_out, self.rtmo_scaler = self.dataLoader(
            dC.hourly_df,
            p.lookback_hourly,
            p.test_portion,
            v.y_rtmo_vars,
            v.x_rtmo_vars,
            v.x_rtmo_pars)

        enba_train_features, enba_train_out, enba_test_features, enba_test_out, self.enba_scaler = self.dataLoader(
            dC.hourly_df,
            p.lookback_hourly,
            p.test_portion,
            v.y_enba_vars,
            v.x_enba_pars,
            v.x_enba_vars)

        bicm_train_features, bicm_train_out, bicm_test_features, bicm_test_out, self.bicm_scaler = self.dataLoader(
            dC.hourly_df,
            p.lookback_hourly,
            p.test_portion,
            v.y_bicm_vars,
            v.x_bicm_vars,
            v.x_bicm_pars)

        rtms_train_features, rtms_train_out, rtms_test_features, rtms_test_out, self.rtms_label_scaler = self.dataLoader(
            dC.hourly_df,
            p.lookback_hourly,
            p.test_portion,
            v.y_rtms_vars,
            v.x_rtms_pars,
            v.x_rtms_vars)

        self.carp_train_dataset = ClassiregressionDataset(carp_train_features, carp_train_out)
        self.rtmo_train_dataset = ClassiregressionDataset(rtmo_train_features, rtmo_train_out)
        self.enba_train_dataset = ClassiregressionDataset(enba_train_features, enba_train_out)
        self.bicm_train_dataset = ClassiregressionDataset(bicm_train_features, bicm_train_out)
        self.rtms_train_dataset = ClassiregressionDataset(rtms_train_features, rtms_train_out)

        self.carp_train_loader = DataLoader(self.carp_train_dataset, shuffle=True, batch_size=p.batch_size_daily,
                                            collate_fn=lambda x: collate_unsuperv(x, max_len=p.lookback_daily), drop_last=True)
        self.rtmo_train_loader = DataLoader(self.rtmo_train_dataset, shuffle=True, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_unsuperv(x, max_len=p.lookback_hourly), drop_last=True)
        self.enba_train_loader = DataLoader(self.enba_train_dataset, shuffle=True, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_unsuperv(x, max_len=p.lookback_hourly), drop_last=True)
        self.bicm_train_loader = DataLoader(self.bicm_train_dataset, shuffle=True, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_unsuperv(x, max_len=p.lookback_hourly), drop_last=True)
        self.rtms_train_loader = DataLoader(self.rtms_train_dataset, shuffle=True, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_unsuperv(x, max_len=p.lookback_hourly), drop_last=True)

        self.carp_test_dataset = ClassiregressionDataset(carp_test_features, carp_test_out)
        self.rtmo_test_dataset = ClassiregressionDataset(rtmo_test_features, rtmo_test_out)
        self.enba_test_dataset = ClassiregressionDataset(enba_test_features, enba_test_out)
        self.bicm_test_dataset = ClassiregressionDataset(bicm_test_features, bicm_test_out)
        self.rtms_test_dataset = ClassiregressionDataset(rtms_test_features, rtms_test_out)

        self.carp_test_loader = DataLoader(self.carp_test_dataset, shuffle=False, batch_size=p.batch_size_daily,
                                            collate_fn=lambda x: collate_unsuperv(x, max_len=p.lookback_daily), drop_last=True)
        self.rtmo_test_loader = DataLoader(self.rtmo_test_dataset, shuffle=False, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_unsuperv(x, max_len=p.lookback_hourly), drop_last=True)
        self.enba_test_loader = DataLoader(self.enba_test_dataset, shuffle=False, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_unsuperv(x, max_len=p.lookback_hourly), drop_last=True)
        self.bicm_test_loader = DataLoader(self.bicm_test_dataset, shuffle=False, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_unsuperv(x, max_len=p.lookback_hourly), drop_last=True)
        self.rtms_test_loader = DataLoader(self.rtms_test_dataset, shuffle=False, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_unsuperv(x, max_len=p.lookback_hourly), drop_last=True)

        self.carp_input_dim = len(v.x_carp_vars + v.x_carp_pars)
        self.rtmo_input_dim = len(v.x_rtmo_vars + v.x_rtmo_pars)
        self.enba_input_dim = len(v.x_enba_vars + v.x_enba_pars)
        self.bicm_input_dim = len(v.x_bicm_vars + v.x_bicm_pars)
        self.rtms_input_dim = len(v.x_rtms_vars + v.x_rtms_pars)

        self.carp_output_dim = len(v.y_carp_vars)
        self.rtmo_output_dim = len(v.y_rtmo_vars)
        self.enba_output_dim = len(v.y_enba_vars)
        self.bicm_output_dim = len(v.y_bicm_vars)
        self.rtms_output_dim = len(v.y_rtms_vars)

    def dataLoader(self, df, breath_steps, test_portion, y_field, x_field_vars, x_field_pars):
        df = reduce_mem_usage(df)

        # Split data into train/test portions and combining all data from different files into a single array
        test_length = int(test_portion * len(df))

        train_features = df.iloc[:-test_length][x_field_vars + x_field_pars]
        test_features = df.iloc[-test_length:][x_field_vars + x_field_pars]

        train_out = df.iloc[:-test_length][y_field]
        test_out = df.iloc[-test_length:][y_field]

        scaler = MinMaxScaler()

        # Fit and transform the selected columns and replace them in the dataframe
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        train_features = train_features[:(train_features.shape[0] // breath_steps) * breath_steps]
        test_features = test_features[:(test_features.shape[0] // breath_steps) * breath_steps]
        train_out = train_out[:(train_out.shape[0] // breath_steps) * breath_steps]
        test_out = test_out[:(test_out.shape[0] // breath_steps) * breath_steps]

        train_features = train_features.reshape(-1, breath_steps, train_features.shape[-1])
        test_features = test_features.reshape(-1, breath_steps, test_features.shape[-1])

        train_out = train_out.to_numpy().reshape(-1, breath_steps, train_out.shape[-1])
        test_out = test_out.to_numpy().reshape(-1, breath_steps, test_out.shape[-1])

        all_features = np.concatenate([train_features, test_features], axis=0)
        all_out = np.concatenate([train_out, test_out], axis=0)

        # return all_features, all_out, scaler
        return train_features, train_out, test_features, test_out, scaler
