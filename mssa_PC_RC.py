# ========================================================
# This code contains the functions for the MSSA workflow
# author: artemis zr
# started: 17/07/2026
# ========================================================


import numpy as np
from scipy.linalg import toeplitz

from sklearn.utils.extmath import randomized_svd



# --- Perform the MSSA on the already embedded dataset

def mssa_workflow(Xemb, pcs, M, type='ECON'):
    N = pcs.shape[0]
    D = pcs.shape[1]
    U_out = None  # sera rempli seulement en BKi

    if type == 'ECON':
        if N-M+1 < D*M: 
            type = 'BKi'
        else: 
            type = 'BK'

    if type == 'BKi':
        C = Xemb.T @ Xemb / (D * M)
        n_lim = min(N - M + 1, D * M)
        U, s, Vh = randomized_svd(Xemb / np.sqrt(N - M + 1), n_components=n_lim)
        eigval  = s**2
        eigvect = Vh.T
        U_out   = U          # shape = (N-M+1, K) (needed for Monte Carlo test)
        expvar  = eigval / eigval.sum() * 100

    if type == 'BK':
        C = Xemb.T @ Xemb / (D * M)
        eigval, eigvect = np.linalg.eigh(C)
        ind     = np.argsort(eigval)[::-1]  # bcs eigh gives eigval in ascending order
        eigval  = eigval[ind]
        eigvect = eigvect[:, ind]
        expvar  = eigval / eigval.sum() * 100

    PCs = Xemb @ eigvect

    return eigvect, eigval, expvar, PCs, U_out  # U_out = None si BK ou eig




# --- Compute the dominant frequency for each PC 

def dominant_frequency(PCs, fs=1):
    """
    fs : int or float, optional, default=1
        Sampling frequency:
        - Use 1 if you have yearly data (this will give frequency in cycles/year)
        - Use 12 if you have monthly data (this will give frequency in cycles/year)
    """
    n_samples, n_modes = PCs.shape

    # Next power of 2 ≥ n_samples (MATLAB logic)
    nfft = 1
    while nfft < n_samples:
        nfft *= 2

    # MATLAB frequency grid
    f_scale = np.arange(0, nfft//2 + 1) * fs / nfft

    dominant_freq = np.zeros(n_modes)
    dominant_period = np.zeros(n_modes)

    for j in range(n_modes):
        #x = PCs[:, j]             # do NOT remove mean
        x = PCs[:, j] - np.mean(PCs[:, j])


        y = np.fft.fft(x, nfft)   # zero-padded FFT
        power = np.abs(y)**2      # MATLAB: abs(y.^2)
        power = power[:nfft//2 + 1]

        k = np.argmax(power)
        dominant_freq[j] = f_scale[k]
        dominant_period[j] = 1.0 / dominant_freq[j] if dominant_freq[j] != 0 else np.inf

    return dominant_freq, dominant_period



# --- Compute the reconstructed components RCs (taken from the Matlab code from Gaston Manta)

def reconstruct_components(EV, AA, n, M, N, D):
    """
    Parameters:
    EV : ndarray, shape (M*D, K)
        Eigenvectors AFTER projection (Y' * RHO_original)
    AA : ndarray, shape (N-M+1, K)
        Principal components A = Y * EV
    n  : Indices of modes to reconstruct
    M  : Embedding window length
    N  : Original time series length
    D  : Number of channels

    Returns:
    RC : ndarray, shape (len(n), N, D) = (mode, time, channel)
        Reconstructed components
    """

    K = len(n)
    RC = np.zeros((K, N, D))

    # --- Normalize eigenvectors ---
    EV = EV.copy()
    for k in range(EV.shape[1]):
        EV[:, k] /= np.linalg.norm(EV[:, k])

    for kk, k in enumerate(n):

        # MATLAB: ek = reshape(RHO_full(:,k), M, D)
        ek = EV[:, k].reshape(M, D, order='F')

        for t in range(N):  # MATLAB n = 1:N

            if t <= M - 2:
                Mn = t + 1
                L = 0
                U = t + 1
            elif t <= N - M:
                Mn = M
                L = 0
                U = M
            else:
                Mn = N - t
                L = t - (N - M)
                U = M

            for d in range(D):
                acc = 0.0
                for m in range(L, U):
                    acc += AA[t - m, k] * ek[m, d]
                RC[kk, t, d] = acc / Mn
    RC = np.transpose(RC, (1, 2, 0))  # (K, N, D) → (N, D, K)

    return RC



# --- Separate of the RCs by variables

def separate_RCs_by_var(RCs,n_vars,D_eof):
    RCs_by_var = []
    for ivar in range(n_vars):
        start = ivar * D_eof
        end   = (ivar + 1) * D_eof
        RCs_by_var.append(RCs[:, start:end, :]) # RCs_by_var[i].shape = (N, D_eof, K=number of modes)
    return RCs_by_var



# --- Put RCs back in the lat/lon domain (grouped by pairs or by selection)

def reconstruct_RCs_group(
    RCs_var,        # (N, D_eof, K)
    eofs_maps,      # (D_eof, lat, lon)
    std_field,      # (lat, lon)
    rc_indices      # list of RC indices, 1-based
):
    """
    Parameters:
    RCs_var : np.ndarray, shape (N, D_eof, K), RCs for ONE variable, with K being the number of modes from the MSSA 
    eofs_maps : xarray.DataArray, shape (D_eof, lat, lon)
    std_field : Oeiginal field (lat, lon)
    rc_indices : RC numbers to reconstruct (1-based, e.g. [1], [2,3], [2,3,4])

    Returns:
    VAR_RC : np.ndarray, reconstructed field (lat, lon, N)
    rc_map_mean : np.ndarray, time-mean spatial pattern (lat, lon)
    rc_timeserie : np.ndarray, area-mean time series (N,)
    """

    # --- Dimensions
    N, D_eof, K = RCs_var.shape
    lat_len, lon_len = eofs_maps.shape[1:]

    # --- Prepare matrices
    RCs_var_T = np.transpose(RCs_var, (1, 0, 2))  # (D_eof, N, K)
    eofs_2d = eofs_maps.values.reshape(D_eof, -1).T  # (lat*lon, D_eof)

    # --- Convert RC indices (1-based → 0-based)
    rc_idx = [i - 1 for i in rc_indices]

    # --- Sum selected RCs
    rc_sum = np.sum(RCs_var_T[:, :, rc_idx], axis=2)  # (D_eof, N)

    # --- Back to physical space
    VAR_RC = eofs_2d @ rc_sum   # (lat*lon, N)
    VAR_RC = VAR_RC.reshape(lat_len, lon_len, N)
    std_field_np = std_field.values if hasattr(std_field, "values") else std_field
    VAR_RC *= std_field_np[:, :, np.newaxis]

    # --- Diagnostics
    rc_map_mean = VAR_RC.mean(axis=2)
    rc_timeserie = np.nanmean(VAR_RC, axis=(0, 1))

    return VAR_RC, rc_timeserie, rc_map_mean

# --- Example for the calculation of RC1 for the var1
# --- If I want to reconstruct RC1 for the var2, I need to change RCs_by_var[1], eofs_maps[1] and var_std[1] 
#VAR1_RC1, _ = reconstruct_RCs_group(
#    RCs_by_var[0],
#    eofs_maps[0],
#    var_std[0],
#    [1]
#)


# --- To build the different phases in the RCs plots

def compute_mssa_phase_from_RC_field(VAR_RC):
    """
    VAR_RC: ndarray (time, lat, lon), reconstructed RC field (or sum of RCs)
    Returns:
        phin: instantaneous phase (time,)
    """

    # --- reshape to (time, space)
    T, ny, nx = VAR_RC.shape
    ff = VAR_RC.reshape(T, ny * nx)

    # --- remove NaN grid points
    valid = ~np.isnan(ff).any(axis=0)
    ff = ff[:, valid]

    # --- append last column (Matlab trick)
    ff = np.concatenate([ff, ff[:, -1][:, None]], axis=1)

    # --- SVD
    U, S, Vt = np.linalg.svd(ff, full_matrices=False)

    # --- instantaneous phase
    phin = np.angle(U[:, 0] + 1j * U[:, 1])

    return phin


def phase_indices_from_phin(phin, n_phases=4):
    phases = []
    for p in range(1, n_phases + 1):
        lower = (p - 1) * np.pi / 2 - np.pi
        upper = p * np.pi / 2 - np.pi
        idx = np.where((phin > lower) & (phin <= upper))[0]
        phases.append(idx)
    return phases


