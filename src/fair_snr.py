"""
Fair, multi-perspective SNR and quality metrics for DAS denoising evaluation.

Implements THREE complementary families of metrics (addressing Reviewer 3/4
concerns about image-metric bias):

Family A — Semblance-based local SNR (the metric used in the original paper,
           but now with TWO variants):
  A1) `semblance_snr_aligned`   — WITH inter-trace cross-correlation alignment
                                  (original implementation, can favor
                                   continuity-promoting methods)
  A2) `semblance_snr_unaligned` — WITHOUT alignment (stricter, avoids bias)

Family B — Energy-based standard SNR (geophysical standard definition):
  B1) `energy_snr_manual_window` — SNR = 10*log10( signal_energy / noise_energy )
                                  based on manual signal vs noise time windows

Family C — Physical-fidelity metrics (on synthetic data only):
  C1) `arrival_time_error`        — Δt of STA/LTA picks vs ground truth
  C2) `waveform_correlation`      — Pearson CC between denoised and GT
  C3) `spectral_conformity`       — L2 ratio of amplitude spectra in band

All functions are vectorized / automated so that reviewing readers can
reproduce any number without manual cherry-picking.
"""

import numpy as np
from scipy.signal import correlate, find_peaks
from scipy.ndimage import generic_filter


# ============================================================
# Helpers: Cross-correlation alignment (for Family A1)
# ============================================================
def _cc_align_trace_pair(trace_a, trace_b, cc_thresh=0.9):
    """
    Shift trace_b to align with trace_a by maximizing 1-D cross correlation.
    If max CC < cc_thresh, skip alignment (return original pair, potentially
    padded with zeros on only one side to match length).
    """
    corr = correlate(trace_a, trace_b, mode='full')
    n = len(trace_a)
    lags = np.arange(-(n - 1), n)
    peak = np.argmax(corr)
    lag = lags[peak]
    if abs(lag) >= n - 1:
        return trace_a, trace_b, 0
    # Verify CC at the best lag
    if lag >= 0:
        ta = trace_a[lag:]
        tb = trace_b[:n - lag]
    else:
        ta = trace_a[:n + lag]
        tb = trace_b[-lag:]
    if len(ta) < 5 or len(tb) < 5:
        return trace_a, trace_b, 0
    cc = np.corrcoef(ta, tb, rowvar=False)[0, 1]
    if np.isnan(cc) or cc < cc_thresh:
        return trace_a, trace_b, 0
    # Pad both to original length by zero padding
    if lag >= 0:
        tb_out = np.concatenate([trace_b[:n - lag], np.zeros(lag)])
        ta_out = trace_a.copy()
    else:
        tb_out = np.concatenate([np.zeros(-lag), trace_b[-lag:]])
        ta_out = trace_a.copy()
    return ta_out, tb_out, (1 if cc >= cc_thresh else 0)


def _window_semblance(region, do_align=True, cc_thresh=0.7):
    """
    Compute Marfurt semblance of a (n_traces, n_samples) region.
    Optionally align adjacent traces first (original implementation).
    """
    region = np.asarray(region, dtype=np.float64).reshape(
        region.shape[-2], region.shape[-1])
    ntraces, nsamples = region.shape
    if do_align:
        aligned = [region[0].copy()]
        for i in range(ntraces - 1):
            a = aligned[-1]
            b = region[i + 1].copy()
            _, b_shifted, ok = _cc_align_trace_pair(a, b, cc_thresh=cc_thresh)
            aligned.append(b_shifted)
        region = np.stack(aligned, axis=0)
        ntraces, nsamples = region.shape
    square_of_sums = np.sum(region, axis=0) ** 2
    sum_of_squares = np.sum(region ** 2, axis=0)
    denom = np.sum(sum_of_squares)
    if denom < 1e-12:
        return 0.0
    sembl = np.sum(square_of_sums) / denom
    return sembl / max(ntraces, 1)


def _semblance_map(data, n_traces=13, n_samples=19, do_align=True,
                   cc_thresh=0.7):
    """Compute per-pixel local semblance via moving window."""
    # data shape: (H_time, W_channel).  We want windows over
    # (n_traces channels, n_samples time samples).  generic_filter treats
    # input shape flat, so we transpose data so axis order matches region.
    data_tx = np.ascontiguousarray(data.T)  # (W, H)  traces first
    def _f(flat):
        return _window_semblance(flat.reshape(n_traces, n_samples),
                                 do_align=do_align, cc_thresh=cc_thresh)
    sembl_tx = generic_filter(data_tx, _f, size=(n_traces, n_samples),
                              mode='constant', cval=0.0)
    sembl_ht = sembl_tx.T  # back to (H_time, W_channel)
    return sembl_ht


# ============================================================
# Family A — Semblance-based local SNR (two variants)
# ============================================================
def semblance_based_snr(data, event_time_slices, channel_slice=slice(100, 650),
                        n_traces=13, n_samples=19, fs=1000,
                        aligned=True, cc_thresh=0.7):
    """
    Semblance → local SNR = S / (1 - S), converted to dB via factor 10
    (as in sp2p.tex Eq. 1).

    Parameters:
        data: (H_time, W_channel)
        event_time_slices: list of (start_sample_idx, end_sample_idx) tuples
                           for each manually-labeled event window
        channel_slice: slice object, channels to include (default matches the
                       original code's [100:650])
        aligned: if True, run A1; if False, run A2 (no alignment, stricter)

    Returns:
        snr_db_per_event: list of float (one per event_time_slice)
        snr_map_full: (H, W) array, per-pixel SNR in dB (for visualization)
    """
    S = _semblance_map(data, n_traces=n_traces, n_samples=n_samples,
                       do_align=aligned, cc_thresh=cc_thresh)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = S / np.clip(1.0 - S, 1e-12, None)
        snr_map_db = 10.0 * np.log10(np.clip(ratio, 1e-12, None))

    event_snr = []
    for (t0, t1) in event_time_slices:
        roi = snr_map_db[t0:t1, channel_slice]
        event_snr.append(float(np.nanmean(roi)))
    return event_snr, snr_map_db


# ============================================================
# Family B — Standard Energy-based SNR (geophysical definition)
# ============================================================
def energy_based_snr(data, signal_time_slices, noise_time_slices,
                     channel_slice=slice(100, 650)):
    """
    Canonical geophysical SNR:
        SNR_dB = 10 * log10( mean(|s|^2 in signal window) /
                             mean(|n|^2 in noise window) )

    Signal and noise windows must be supplied by the caller, e.g.:
        signal_time_slices = [(305,400), (450,545)]   # event positions
        noise_time_slices  = [(50,250), (700,950)]     # quiet regions

    Returns per-event SNR_dB (using the SAME noise window for all events),
    plus a per-channel SNR array for visualization.
    """
    noise_parts = []
    for (t0, t1) in noise_time_slices:
        noise_parts.append(data[t0:t1, channel_slice])
    noise = np.concatenate(noise_parts, axis=0)
    noise_energy = np.mean(noise ** 2) + 1e-24

    event_snr = []
    for (t0, t1) in signal_time_slices:
        sig = data[t0:t1, channel_slice]
        sig_energy = np.mean(sig ** 2)
        event_snr.append(float(10.0 * np.log10(sig_energy / noise_energy)))

    return event_snr, noise_energy


# ============================================================
# Family C — Synthetic-data physical-fidelity metrics
# ============================================================
def sta_lta_pick(trace, sta=5, lta=50, threshold=3.0, fs=1000):
    """Simple STA/LTA onset picker. Returns pick sample index or None."""
    trace = np.asarray(trace, dtype=np.float64)
    energy = trace ** 2
    sta = int(sta)
    lta = int(lta)
    if len(trace) <= lta:
        return None
    csum = np.concatenate([[0.0], np.cumsum(energy)])
    n = np.arange(1, len(energy) + 1)
    sta_win = np.minimum(sta, n)
    lta_win = np.minimum(lta, n)
    sta_vals = (csum[n] - csum[np.maximum(n - sta, 0)]) / sta_win
    lta_vals = (csum[n] - csum[np.maximum(n - lta, 0)]) / lta_win
    ratio = sta_vals / (lta_vals + 1e-24)
    peaks, props = find_peaks(ratio, height=threshold, distance=max(lta, sta) * 2)
    if len(peaks) == 0:
        return None
    return int(peaks[0])  # first arrival pick


def arrival_time_error(denoised_trace, gt_trace, fs=1000, **picker_kwargs):
    """Δt in ms. Returns NaN if either pick fails."""
    p_den = sta_lta_pick(denoised_trace, fs=fs, **picker_kwargs)
    p_gt = sta_lta_pick(gt_trace, fs=fs, **picker_kwargs)
    if p_den is None or p_gt is None:
        return float('nan')
    return float((p_den - p_gt) / fs * 1000.0)  # milliseconds


def waveform_correlation(denoised_trace, gt_trace, signal_window=None):
    """Pearson correlation coefficient in the specified window."""
    if signal_window is not None:
        t0, t1 = signal_window
        d = denoised_trace[t0:t1]
        g = gt_trace[t0:t1]
    else:
        d = denoised_trace
        g = gt_trace
    if np.std(d) < 1e-12 or np.std(g) < 1e-12:
        return float('nan')
    return float(np.corrcoef(d, g, rowvar=False)[0, 1])


def spectral_conformity(denoised_trace, gt_trace, fs=1000, band=(30.0, 100.0)):
    """
    Ratio of amplitude spectra within the signal band, measuring how well the
    denoised result preserves the frequency content of ground truth.

    Returns (1 - |1 - ratio|), 1.0 = perfect match.
    """
    from scipy.signal import welch
    f, Pd = welch(denoised_trace, fs=fs, nperseg=min(256, len(denoised_trace)))
    _, Pg = welch(gt_trace, fs=fs, nperseg=min(256, len(gt_trace)))
    mask = (f >= band[0]) & (f <= band[1])
    Ed = np.trapz(Pd[mask], f[mask]) + 1e-24
    Eg = np.trapz(Pg[mask], f[mask]) + 1e-24
    ratio = Ed / Eg
    return float(np.clip(1.0 - abs(1.0 - ratio), 0.0, 1.0))


# ============================================================
# Utility: event windows matching original snr_res_seg*.txt boxes
# ============================================================
def default_event_windows(segment_id):
    """
    Time-sample slices (t0, t1) matching the red boxes in ZSN2N-DAS-SNR.py:
      Segment 1: Event 1 [305:400], Event 2 [450:545]  (0.305-0.400s, 0.450-0.545s)
      Segment 2: Events [270:395], [435:640], [860:965]
    plus two suggested quiet noise windows [50:250], [700:950].
    """
    if segment_id == 1:
        signal = [(305, 400), (450, 545)]
        noise = [(50, 250), (700, 900)]
    elif segment_id == 2:
        signal = [(270, 395), (435, 640), (860, 965)]
        noise = [(50, 240), (660, 840)]
    else:
        raise ValueError("segment_id must be 1 or 2")
    return signal, noise
