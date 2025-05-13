"""This file implements amputation mechanisms (MCAR, MAR (logisitc) and MNAR (logistic)) for missing data generation.
It was inspired from: https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values
Original code: https://github.com/BorisMuzellec/MissingDataOT/blob/master/utils.py
"""

import gin
import torch
import logging
import numpy as np
from scipy import optimize


def fill_missing_with_mean(X: torch.Tensor) -> torch.Tensor:
    """
    Returns a copy of X where all NaNs have been replaced
    with the column-wise mean (computed over non-NaN entries).

    This is to be used ONLY for MAR/MNAR where we build a logistic model
    Such a model requires fully complete data for learning. Note that these
    mean values will *not* show up in the resulting amputated data, its 
    just for learning weights.
    """
    X_filled = X.clone()
    for j in range(X.shape[1]):
        col = X_filled[:, j]
        nan_mask = torch.isnan(col)
        if nan_mask.any():
            mean_val = col[~nan_mask].mean()
            col[nan_mask] = mean_val
    return X_filled

def MCAR_mask(X, p):
    """
    Missing completely at random mechanism.

    Parameters
    ----------
    X : torch.FloatTensor, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : torch.BoolTensor
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape
    mask = np.zeros((n, d))

    ber = torch.rand(n, d)
    mask = ber < p

    return mask


def BO_mask(X, p):
    """
    Black out missing mechanism. Removes values across dimensions.

    Parameters
    ----------
    X : torch.FloatTensor, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : torch.BoolTensor
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    indices = torch.randperm(n)[: int(n * p)]
    mask = torch.zeros(n, d).bool()
    mask[indices, :] = True

    return mask


def extract_imputed_blocks(mask, imputed_stds):
    # Result list to store the imputed blocks
    imputed_blocks = []
    # Iterate over each sample in the batch
    for i in range(mask.shape[0]):
        mask_row = mask[i]
        data_row = imputed_stds[i]
        start_idx = None
        for t in range(mask.shape[1]):
            if mask_row[t] == 1:
                if start_idx is None:
                    start_idx = t
            else:
                if start_idx is not None:
                    block = data_row[start_idx:t]
                    imputed_blocks.append(block)
                    start_idx = None
        
        # Handle the case where the block reaches the end of the time series
        if start_idx is not None:
            block = data_row[start_idx:]
            imputed_blocks.append(block)

    block_sizes = [block.shape[0] for block in imputed_blocks]
    #print("Detected block sizes:", set(block_sizes))
    return imputed_blocks

def blockBO_mask(X, p):
    n, d = X.shape
    total_missing_values = int(n * p * d)
    mask = torch.zeros(n, d).bool()
    occupied_indices = set()

    #block_sizes = [2, 4, 8, 16, 32]
    block_sizes = [2, 3, 4, 5, 6]
    current_missing_count = 0

    max_attempts = n * 10  # To prevent infinite loops
    attempts = 0

    while current_missing_count < total_missing_values and attempts < max_attempts:
        start_idx = torch.randint(0, n, (1,)).item()
    
        # Ensure that the block fits within the data bounds
        possible_block_sizes = [size for size in block_sizes if start_idx + size <= n]
        possible_block_sizes = [block_size for block_size in possible_block_sizes if block_size <= total_missing_values-current_missing_count]
        if len(possible_block_sizes) == 0:
            attempts += 1
            continue
        
        #print('possible_block_sizes: ', possible_block_sizes)
        if not possible_block_sizes:
            attempts += 1
        #    print('Skipping 1')
            continue

        valid_block_sizes = []
        for size in possible_block_sizes:
            # Indices to check: start_idx -1, block indices, end_idx (for adjacency)
            indices_to_check = list(range(start_idx - 1 if start_idx > 0 else start_idx, start_idx + size + 1))
            # Ensure all indices to check are within bounds
            indices_to_check = [idx for idx in indices_to_check if 0 <= idx < n]
            if all(idx not in occupied_indices for idx in indices_to_check):
                valid_block_sizes.append(size)

        if not valid_block_sizes:
            attempts += 1
            #print('Skipping 2')
            continue
        #print('valid_block_sizes: ', valid_block_sizes)

        # Randomly select a valid block size
        block_size = valid_block_sizes[torch.randint(0, len(valid_block_sizes), (1,)).item()]
        block_missing_values = block_size * d

        if current_missing_count + block_missing_values > total_missing_values:
            attempts += 1
            #print("Skipping 3")
            continue

        end_idx = start_idx + block_size
        #print('end_idx: ', end_idx)
        # Update mask and occupied indices
        mask[start_idx:end_idx, :] = True
        
        # Mark the block and its adjacent indices as occupied
        indices_to_mark = list(range(start_idx - 1 if start_idx > 0 else start_idx, end_idx + 1))
        indices_to_mark = [idx for idx in indices_to_mark if 0 <= idx < n]
        occupied_indices.update(indices_to_mark)

        current_missing_count += block_missing_values
        attempts = 0  # Reset attempts after successful placement

    if attempts >= max_attempts:
        print("Warning: Maximum attempts reached while placing blocks.")
    extract_imputed_blocks(mask.unsqueeze(0)[:, :, 0], torch.randn_like(mask.unsqueeze(0)[:, :, 0].float()))
    return mask


'''
def blockBO_mask(X, p):
    T, d = X.shape
    total_values = T * d
    target_missing_values = int(p * total_values)
    target_missing_timesteps = int(np.ceil(target_missing_values / d))

    mask = torch.zeros((T, d)).bool()
    unmasked_intervals = [(0, T - 1)]  # List of intervals where data is unmasked
    total_masked_timesteps = 0
    block_lengths = [2, 4, 8, 16, 32]

    while total_masked_timesteps < target_missing_timesteps and unmasked_intervals:
        L = np.random.choice(block_lengths)

        # Find intervals where L can fit
        intervals_with_L = [(a, b) for (a, b) in unmasked_intervals if (b - a + 1) >= L]

        if not intervals_with_L:
            continue

        # Collect possible start positions and corresponding interval indices
        possible_starts = []
        interval_indices = []
        for idx, (a, b) in enumerate(intervals_with_L):
            starts = list(range(a, b - L + 2))  # possible start positions
            possible_starts.extend(starts)
            interval_indices.extend([idx] * len(starts))

        if not possible_starts:
            continue

        s_idx = np.random.randint(len(possible_starts))
        s = possible_starts[s_idx]
        interval_idx_in_intervals_with_L = interval_indices[s_idx]
        (a, b) = intervals_with_L[interval_idx_in_intervals_with_L]

        # Find the index of (a, b) in unmasked_intervals
        interval_idx_in_unmasked = unmasked_intervals.index((a, b))

        # Update unmasked intervals
        new_intervals = []
        if a < s:
            new_intervals.append((a, s - 1))
        if s + L <= b:
            new_intervals.append((s + L, b))

        # Replace the interval with new intervals
        unmasked_intervals.pop(interval_idx_in_unmasked)
        for ni in reversed(new_intervals):
            unmasked_intervals.insert(interval_idx_in_unmasked, ni)

        # Update mask
        end = s + L
        mask[s:end, :] = True
        total_masked_timesteps += L


    return mask
'''


def MAR_logistic_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : torch.FloatTensor, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape
    mask = torch.zeros(n, d).bool()

    # number of variables that will have no missing values (at least one variable)
    d_obs = max(int(p_obs * d), 1)
    # number of variables that will have missing values
    d_na = d - d_obs

    # Sample variables that will all be observed, and those with missing values
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # Other variables will have NA proportions that depend on those observed variables, through a logistic model
    # The parameters of this logistic model are random

    X_filled = fill_missing_with_mean(X)
    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X_filled, idxs_obs, idxs_nas)
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X_filled[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X_filled[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


def MNAR_logistic_mask(X, p, p_params=0.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

    Parameters
    ----------
    X : torch.FloatTensor, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)

    Returns
    -------
    mask : torch.BoolTensor
        Mask of generated missing values (True if the value is missing).
    """
    
    n, d = X.shape
    mask = torch.zeros(n, d).bool()

    # number of variables used as inputs (at least 1)
    d_params = max(int(p_params * d), 1) if exclude_inputs else d
    # number of variables masked with the logistic model
    d_na = d - d_params if exclude_inputs else d

    # Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    # Other variables will have NA proportions selected by a logistic model
    # The parameters of this logistic model are random.

    X_filled = fill_missing_with_mean(X)
    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X_filled, idxs_params, idxs_nas)
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X_filled[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X_filled[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    # If the inputs of the logistic model are excluded from MNAR missingness, mask some
    # values used in the logistic model at random
    # This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None):
    d_obs = len(idxs_obs)
    d_na = len(idxs_nas)
    coeffs = torch.randn(d_obs, d_na)
    Wx = X[:, idxs_obs].mm(coeffs)
    coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p):
    d_obs, d_na = coeffs.shape
    intercepts = torch.zeros(d_na)
    for j in range(d_na):

        def f(x):
            return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

        intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


@gin.configurable("amputation")
def ampute_data(data, mechanism, p_miss, p_obs=0.3):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values.

    Parameters
    ----------
    data : DataFrame
        Data for which missing values will be simulated.
    mechanism : str,
            Indicates the missing-data mechanism to be used. ("MCAR", "MAR" or "MNAR")
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
            If mecha = "MAR" or "MNAR", proportion of variables with *no* missing values
            that will be used for the logistic masking model.

    Returns
    ----------
    imputed_data: DataFrame
        The data with the generated missing values.
    """
    logging.info(f"Applying {mechanism} amputation.")
    X = torch.tensor(data.values.astype(np.float32))

    if mechanism == "MAR":
        mask = MAR_logistic_mask(X, p_miss, p_obs)
    elif mechanism == "MNAR":
        mask = MNAR_logistic_mask(X, p_miss, p_obs)
    elif mechanism == "MCAR":
        mask = MCAR_mask(X, p_miss)
    elif mechanism == "BO":
        mask = BO_mask(X, p_miss)
    elif mechanism == "blockBO":
        mask = blockBO_mask(X, p_miss)
    else:
        logging.error("Not a valid amputation mechanism. Missing-data mechanisms to be used are MCAR, MAR or MNAR.")

    amputed_data = data.mask(mask)
    return amputed_data, mask
