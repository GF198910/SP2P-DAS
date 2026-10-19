"""
Independent preprocessing module for DAS denoising.
All steps are decoupled and can be switched on/off individually.
This module is designed for transparency and ablation study.

Author: Codebase restructuring for SP2P-DAS paper revision
"""

import numpy as np
from scipy.signal import butter, filtfilt


# ============================================================
# 1. Bandpass Filter
# ============================================================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut=30.0, highcut=100.0, fs=1000, order=5):
    """
    Apply zero-phase Butterworth bandpass filter along the time axis (axis=0).

    Parameters:
        data: np.ndarray (H_time, W_channel)
        lowcut: lower cutoff frequency in Hz
        highcut: upper cutoff frequency in Hz
        fs: sampling frequency in Hz
        order: filter order

    Returns:
        filtered: np.ndarray same shape as data
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = filtfilt(b, a, data, axis=0)
    # filtfilt can return negative-stride arrays ([::-1] flips inside);
    # torch.from_numpy rejects them -> force contiguous copy
    return np.ascontiguousarray(filtered)


# ============================================================
# 2. Weak-amplitude Nonlinear Suppression
# ============================================================
def weak_amplitude_suppression(data, sigma_scale=2.0, divisor=5.0):
    """
    Suppress samples with amplitude below sigma_scale * std.
    Originally used in pixel2pixel.py / zsn2n.py but NOT disclosed in paper.

    Parameters:
        data: np.ndarray
        sigma_scale: threshold multiplier on global std
        divisor: values < threshold are divided by this

    Returns:
        processed: np.ndarray
    """
    threshold = np.std(data) * sigma_scale
    processed = data.copy()
    mask = np.abs(processed) < threshold
    processed[mask] /= divisor
    return processed


# ============================================================
# 3. Border Attenuation (Edge Division)
# ============================================================
def apply_border_attenuation(matrix, border_time=20, border_channel=50,
                             divisor=10.0):
    """
    Divide border regions by divisor to suppress edge artifacts.
    Originally hardcoded: 20 time samples (top/bottom), 50 channels (left/right)

    Parameters:
        matrix: np.ndarray (H_time, W_channel)
        border_time: number of time samples to attenuate on top & bottom
        border_channel: number of channels to attenuate on left & right
        divisor: attenuation factor

    Returns:
        processed: np.ndarray
    """
    rows, cols = matrix.shape
    processed = matrix.copy()

    # Top border (first border_time rows, middle columns)
    if border_time > 0 and border_channel > 0:
        processed[:border_time, border_channel:-border_channel] /= divisor
        # Bottom border (last border_time rows, middle columns)
        processed[-border_time:, border_channel:-border_channel] /= divisor
    return processed


# ============================================================
# 4. Channel-wise Z-score Normalization (per-channel)
# ============================================================
def channel_zscore(data):
    """
    Per-channel (column-wise) z-score standardization.
    Returns normalized data plus statistics for inverse transform.
    """
    mean = np.mean(data, axis=0, keepdims=True)   # (1, W)
    std = np.std(data, axis=0, keepdims=True)     # (1, W)
    std_safe = np.where(std < 1e-8, 1.0, std)
    normalized = (data - mean) / std_safe
    return normalized, mean, std_safe


def channel_zscore_inverse(normalized, mean, std):
    """Inverse of channel_zscore. Note: mean optionally dropped (as in original code)."""
    return normalized * std  # original code multiplies std back but NOT adds mean


# ============================================================
# 5. Resize to 64-multiples (bilinear interpolation)
# ============================================================
import torch
import torch.nn.functional as F


def resize_to_multiple(data, multiple=64, target_size=None):
    """
    Resize data so that dimensions are multiples of `multiple` (for block-wise
    patch processing). If target_size is given, resize to that directly.

    Returns:
        resized: torch.Tensor (1,1,H',W') on CPU float32
        original_hw: (H,W)
        flag_resize: bool
    """
    matH, matW = data.shape
    if target_size is not None:
        th, tw = target_size
    else:
        th = ((matH + multiple - 1) // multiple) * multiple
        tw = ((matW + multiple - 1) // multiple) * multiple

    if th == matH and tw == matW:
        flag_resize = False
        tensor = torch.from_numpy(data).float()[None, None, :, :]
        return tensor, (matH, matW), flag_resize
    else:
        flag_resize = True
        tensor = torch.from_numpy(data).float()[None, None, :, :]
        resized = F.interpolate(tensor, size=(th, tw), mode='bilinear',
                                align_corners=False)
        return resized, (matH, matW), flag_resize


def resize_back(tensor, original_hw):
    """Inverse of resize_to_multiple."""
    H, W = original_hw
    back = F.interpolate(tensor, size=(H, W), mode='bilinear',
                         align_corners=False)
    return back.squeeze(0).squeeze(0).cpu().numpy()


# ============================================================
# 6. Full Preprocessing Pipeline (with flags)
# ============================================================
class Preprocessor:
    """
    Configurable preprocessor. Each step can be switched on/off.

    Usage:
        prep = Preprocessor(
            do_bandpass=True, lowcut=30, highcut=100, fs=1000,
            do_weak_suppress=True, sigma_scale=2.0, divisor=5.0,
            do_border_atten=True, border_time=20, border_channel=50,
            do_zscore=True,
            do_resize=True, multiple=64
        )
        tensor, aux = prep(data)
    """

    def __init__(self,
                 do_bandpass=True, lowcut=30.0, highcut=100.0, fs=1000, order=5,
                 do_weak_suppress=True, sigma_scale=2.0, weak_divisor=5.0,
                 do_border_atten=True, border_time=20, border_channel=50,
                 atten_divisor=10.0,
                 do_zscore=True,
                 do_resize=True, resize_multiple=64, target_size=None):

        self.do_bandpass = do_bandpass
        self.lowcut, self.highcut, self.fs, self.order = lowcut, highcut, fs, order

        self.do_weak_suppress = do_weak_suppress
        self.sigma_scale = sigma_scale
        self.weak_divisor = weak_divisor

        self.do_border_atten = do_border_atten
        self.border_time = border_time
        self.border_channel = border_channel
        self.atten_divisor = atten_divisor

        self.do_zscore = do_zscore

        self.do_resize = do_resize
        self.resize_multiple = resize_multiple
        self.target_size = target_size

    def __call__(self, data):
        """
        Parameters:
            data: np.ndarray (H_time, W_channel) raw DAS data
        Returns:
            tensor: torch.Tensor (1,1,H',W') ready for neural network input
            aux: dict containing all info needed for post-processing:
                 - 'original_hw'
                 - 'flag_resize'
                 - 'zscore_mean' / 'zscore_std' (if applicable)
        """
        x = data.copy()
        aux = {'zscore_mean': None, 'zscore_std': None,
               'original_hw': data.shape, 'flag_resize': False}

        # Step 1: Z-score per channel (BEFORE bandpass, same as original code)
        if self.do_zscore:
            x, m, s = channel_zscore(x)
            aux['zscore_mean'] = m
            aux['zscore_std'] = s

        # Step 2: Bandpass
        if self.do_bandpass:
            x = bandpass_filter(x, self.lowcut, self.highcut, self.fs, self.order)

        # Step 3: Weak amplitude suppression
        if self.do_weak_suppress:
            x = weak_amplitude_suppression(x, self.sigma_scale, self.weak_divisor)

        # Step 4: Border attenuation
        if self.do_border_atten:
            x = apply_border_attenuation(x, self.border_time, self.border_channel,
                                         self.atten_divisor)

        # Step 5: Resize
        if self.do_resize:
            tensor, orig_hw, flag = resize_to_multiple(
                x, self.resize_multiple, self.target_size)
            aux['original_hw'] = orig_hw
            aux['flag_resize'] = flag
        else:
            tensor = torch.from_numpy(x).float()[None, None, :, :]

        return tensor, aux

    def postprocess(self, output_tensor, aux, add_mean_back=False):
        """
        Reverse resize + zscore.
        output_tensor: torch.Tensor (1,1,H',W') from network
        """
        # Resize back
        if aux['flag_resize']:
            x = resize_back(output_tensor, aux['original_hw'])  # (H,W) np
        else:
            x = output_tensor.squeeze(0).squeeze(0).cpu().numpy()

        # Inverse z-score (original code does NOT add mean back, only * std)
        if self.do_zscore and aux['zscore_std'] is not None:
            std = aux['zscore_std'].reshape(1, -1)  # (1, W)
            x = x * std
            if add_mean_back and aux['zscore_mean'] is not None:
                mean = aux['zscore_mean'].reshape(1, -1)
                x = x + mean
        return x
