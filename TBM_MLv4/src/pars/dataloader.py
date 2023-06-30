import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset

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

def collate_superv(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks

class Dataloader:
    def __init__(self, dC, v, p):

        carp_X_train, carp_y_train, carp_X_test, carp_y_test, self.carp_label_scaler = self.dataLoader_daily(
            dC.daily_df,
            p.lookback_daily,
            p.test_portion,
            v.y_carp_vars,
            v.x_carp_vars,
            v.x_carp_pars)

        rtmo_X_train, rtmo_y_train, rtmo_X_test, rtmo_y_test, self.rtmo_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            p.lookback_hourly,
            p.test_portion,
            v.y_rtmo_vars,
            v.x_rtmo_vars,
            v.x_rtmo_pars)

        enba_X_train, enba_y_train, enba_X_test, enba_y_test, self.enba_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            p.lookback_hourly,
            p.test_portion,
            v.y_enba_vars,
            v.x_enba_pars,
            v.x_enba_vars)

        bicm_X_train, bicm_y_train, bicm_X_test, bicm_y_test, self.bicm_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            p.lookback_hourly,
            p.test_portion,
            v.y_bicm_vars,
            v.x_bicm_vars,
            v.x_bicm_pars)

        rtms_X_train, rtms_y_train, rtms_X_test, rtms_y_test, self.rtms_label_scaler = self.dataLoader_hourly(
            dC.hourly_df,
            p.lookback_hourly,
            p.test_portion,
            v.y_rtms_vars,
            v.x_rtms_pars,
            v.x_rtms_vars)

        carp_train_data = TensorDataset(torch.from_numpy(carp_X_train), torch.from_numpy(carp_y_train))
        rtmo_train_data = TensorDataset(torch.from_numpy(rtmo_X_train), torch.from_numpy(rtmo_y_train))
        enba_train_data = TensorDataset(torch.from_numpy(enba_X_train), torch.from_numpy(enba_y_train))
        bicm_train_data = TensorDataset(torch.from_numpy(bicm_X_train), torch.from_numpy(bicm_y_train))
        rtms_train_data = TensorDataset(torch.from_numpy(rtms_X_train), torch.from_numpy(rtms_y_train))

        carp_test_data = TensorDataset(torch.from_numpy(carp_X_test), torch.from_numpy(carp_y_test))
        rtmo_test_data = TensorDataset(torch.from_numpy(rtmo_X_test), torch.from_numpy(rtmo_y_test))
        enba_test_data = TensorDataset(torch.from_numpy(enba_X_test), torch.from_numpy(enba_y_test))
        bicm_test_data = TensorDataset(torch.from_numpy(bicm_X_test), torch.from_numpy(bicm_y_test))
        rtms_test_data = TensorDataset(torch.from_numpy(rtms_X_test), torch.from_numpy(rtms_y_test))

        self.carp_train_loader = DataLoader(carp_train_data, shuffle=True, batch_size=p.batch_size_daily,
                                            collate_fn=lambda x: collate_superv(x, max_len=p.lookback_daily), drop_last=True)
        self.rtmo_train_loader = DataLoader(rtmo_train_data, shuffle=True, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_superv(x, max_len=p.lookback_hourly), drop_last=True)
        self.enba_train_loader = DataLoader(enba_train_data, shuffle=True, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_superv(x, max_len=p.lookback_hourly), drop_last=True)
        self.bicm_train_loader = DataLoader(bicm_train_data, shuffle=True, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_superv(x, max_len=p.lookback_hourly), drop_last=True)
        self.rtms_train_loader = DataLoader(rtms_train_data, shuffle=True, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_superv(x, max_len=p.lookback_hourly), drop_last=True)

        self.carp_test_loader = DataLoader(carp_test_data, shuffle=False, batch_size=p.batch_size_daily,
                                            collate_fn=lambda x: collate_superv(x, max_len=p.lookback_daily), drop_last=True)
        self.rtmo_test_loader = DataLoader(rtmo_test_data, shuffle=False, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_superv(x, max_len=p.lookback_hourly), drop_last=True)
        self.enba_test_loader = DataLoader(enba_test_data, shuffle=False, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_superv(x, max_len=p.lookback_hourly), drop_last=True)
        self.bicm_test_loader = DataLoader(bicm_test_data, shuffle=False, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_superv(x, max_len=p.lookback_hourly), drop_last=True)
        self.rtms_test_loader = DataLoader(rtms_test_data, shuffle=False, batch_size=p.batch_size_hourly,
                                            collate_fn=lambda x: collate_superv(x, max_len=p.lookback_hourly), drop_last=True)

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

    def dataLoader_daily(self, df, lookback, test_portion, y_field, x_field_vars, x_field_pars):
        df = df[y_field + x_field_vars + x_field_pars]

        # Scaling the input data
        scaler = MinMaxScaler()
        label_scaler = MinMaxScaler()

        # Fit and transform the selected columns and replace them in the dataframe
        df_fit = df.copy()
        df_fit[y_field + x_field_vars] = scaler.fit_transform(df[y_field + x_field_vars])
        daily_data = df_fit.values
        label_scaler.fit(df[y_field].values.reshape(-1, len(y_field)))

        # Define lookback period and split inputs/labels
        inputs = np.zeros((len(daily_data) - lookback, lookback, len(x_field_vars + x_field_pars)))
        labels = np.zeros((len(daily_data) - lookback, len(y_field)))

        for i in range(lookback, len(daily_data)):
            inputs[i - lookback] = daily_data[i - lookback:i, len(y_field):]
            labels[i - lookback] = daily_data[i, 0:len(y_field)]

        # Split data into train/test portions and combining all data from different files into a single array
        test_length = int(test_portion * len(inputs))

        X_train = inputs[:-test_length]
        y_train = labels[:-test_length]

        X_test = inputs[-test_length:]
        y_test = labels[-test_length:]

        return X_train, y_train, X_test, y_test, label_scaler

    def dataLoader_hourly(self, df, lookback, test_portion, y_field, x_field_vars, x_field_pars):
        df = df[y_field + x_field_vars + x_field_pars]

        # Scaling the input data
        scaler = MinMaxScaler()
        label_scaler = MinMaxScaler()

        # Fit and transform the selected columns and replace them in the dataframe
        df_fit = df.copy()
        df_fit[y_field + x_field_vars] = scaler.fit_transform(df[y_field + x_field_vars])
        hourly_data = df_fit.values
        label_scaler.fit(df[y_field].values.reshape(-1, len(y_field)))

        inputs = np.zeros((len(hourly_data) - lookback, lookback, len(x_field_vars + x_field_pars)))
        labels = np.zeros((len(hourly_data) - lookback, len(y_field)))

        for i in range(lookback, len(hourly_data)):
            inputs[i - lookback] = hourly_data[i - lookback:i, len(y_field):]
            labels[i - lookback] = hourly_data[i, 0:len(y_field)]

        # Split data into train/test portions and combining all data from different files into a single array
        test_length = int(test_portion * len(inputs))

        X_train = inputs[:-test_length]
        y_train = labels[:-test_length]

        X_test = inputs[-test_length:]
        y_test = labels[-test_length:]

        return X_train, y_train, X_test, y_test, label_scaler
