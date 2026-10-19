"""
Structure-Guided Filtering (Structure-Oriented Smoothing) for DAS data.

Independent implementation based on:
  - Hale, D. (2009). Structure-oriented smoothing and semblance.
  - Fehmers & Hocker (2003). Fast structural interpretation with structure
    tensor filtering.
  - Velis (2017). Structure-tensor-based seismic image enhancement.

This module is decoupled from SP2P so it can be (a) used as a standalone
baseline, (b) analyzed in ablation studies, (c) parameterized physically.

Physical interpretation of parameters (for DAS with fs=1000 Hz, ~10m spacing):
  tensor_window : (size_t, size_x) -- e.g. (5,5) ~ (5ms, 50m)
  smoothing_alpha : balance between noise removal (large) and edge fidelity
                    (small). In range [0, 1].
"""

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude


# ============================================================
# Core: Structure Tensor Estimation
# ============================================================
def compute_structure_tensor(data, sigma_deriv=1.0, sigma_window=2.0,
                             window_size=None):
    """
    Compute the 2D structure tensor at every pixel.

    For a scalar image I(t,x):
        J = [  <(dt I)^2>    <dt I dx I>
              <dt I dx I>    <(dx I)^2>  ]
    where <.> denotes Gaussian windowed averaging.

    Parameters:
        data: np.ndarray (H_time, W_channel)
        sigma_deriv: std for Gaussian derivative (gradient smoothing)
        sigma_window: std for structure tensor outer-product smoothing
                      (ignored if window_size is given)
        window_size: (T, X) tuple for uniform window instead of Gaussian
    Returns:
        J11, J12, J22: three 2D arrays (same shape as data)
    """
    from scipy.ndimage import convolve1d
    # Centered finite-difference derivatives (accurate & stable)
    k = np.array([-0.5, 0.0, 0.5])
    It = convolve1d(data.astype(np.float64), k, axis=0, mode='reflect')
    Ix = convolve1d(data.astype(np.float64), k, axis=1, mode='reflect')
    # Gaussian-smooth the partial derivatives before outer product
    if sigma_deriv and sigma_deriv > 0:
        It = gaussian_filter(It, sigma=sigma_deriv)
        Ix = gaussian_filter(Ix, sigma=sigma_deriv)

    # Outer products
    J11_raw = It * It
    J12_raw = It * Ix
    J22_raw = Ix * Ix

    # Windowed averaging
    if window_size is not None:
        from scipy.ndimage import uniform_filter
        wt, wx = window_size
        J11 = uniform_filter(J11_raw, size=(wt, wx), mode='reflect')
        J12 = uniform_filter(J12_raw, size=(wt, wx), mode='reflect')
        J22 = uniform_filter(J22_raw, size=(wt, wx), mode='reflect')
    else:
        J11 = gaussian_filter(J11_raw, sigma=sigma_window)
        J12 = gaussian_filter(J12_raw, sigma=sigma_window)
        J22 = gaussian_filter(J22_raw, sigma=sigma_window)

    return J11, J12, J22


# ============================================================
# Eigenvalue Decomposition -> Orientation, Coherence
# ============================================================
def decompose_structure_tensor(J11, J12, J22):
    """
    Decompose 2x2 structure tensor at every pixel.

    Returns:
        theta: orientation angle (radians) of the MINIMUM curvature direction,
               i.e. the direction along which to smooth (wavefront strike)
        coherence: [0,1] structural coherence = (lam1 - lam2)/(lam1 + lam2)
                   high -> sharp linear structure
        lam1, lam2: eigenvalues (lam1 >= lam2)
    """
    # Trace and determinant for 2x2 eigenvalues
    trace = J11 + J22
    det = J11 * J22 - J12 * J12
    disc = np.sqrt(np.maximum(trace * trace / 4.0 - det, 0.0))
    lam1 = trace / 2.0 + disc  # larger eigenvalue
    lam2 = trace / 2.0 - disc  # smaller eigenvalue

    # Orientation: eigenvector corresponding to lam2 (minimum-variation direction)
    # Solve (J11 - lam2) * v1 + J12 * v2 = 0  ->  v1/v2 = -J12 / (J11 - lam2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(J11 - lam2) > 1e-12,
                         -J12 / (J11 - lam2 + 1e-12),
                         np.where(np.abs(J12) > 1e-12,
                                  -(J22 - lam2) / (J12 + 1e-12),
                                  0.0))
    theta = np.arctan(ratio)  # in (-pi/2, pi/2)

    # Coherence (structure sharpness)
    denom = lam1 + lam2
    coherence = np.where(denom > 1e-12, (lam1 - lam2) / (denom + 1e-12), 0.0)

    return theta, coherence, lam1, lam2


# ============================================================
# Directional Smoothing along wavefront orientation
# ============================================================
def directional_smooth(data, theta, sigma_parallel=3.0, sigma_perpendicular=0.5,
                       coherence=None, alpha_fixed=None):
    """
    Apply anisotropic Gaussian smoothing along the orientation `theta`.

    Strategy: rotate coordinate system by -theta, apply separable Gaussian
    with (sigma_parallel, sigma_perpendicular) along rotated axes,
    rotate back.

    Parameters:
        data: (H,W) image
        theta: (H,W) orientation in radians (minimum-variation direction)
        sigma_parallel: smoothing std ALONG wavefront (direction of coherence)
        sigma_perpendicular: smoothing std ACROSS wavefront (should be small
                             to preserve edges)
        coherence: optional (H,W) array in [0,1]; used to adapt smoothing
                   strength: effective_sigma_parallel *= (coherence * alpha + (1-alpha))
        alpha_fixed: if provided (scalar in [0,1]), overrides adaptive rule;
                     final smoothing = alpha_fixed * full_smooth + (1-alpha) * original
                     (this matches SP2P paper's "alpha=0.8")

    Returns:
        smoothed: (H,W)
    """
    # For efficiency, use steerable filter approximation:
    #   Smoothed = w * orient_smooth + (1-w) * original
    # where orient_smooth is computed via rotated Gaussian kernel.
    # Simpler implementation: for each pixel, we directly sample along the line

    H, W = data.shape

    # Build sampling grid
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Direction vectors along theta (unit vector parallel to wavefront)
    # theta is angle from time axis: direction = (sin, cos) rotated
    # Actually theta is slope of min-variation eigenvector:
    #   v = (cos theta, sin theta) is the wavefront direction
    dx_par = np.cos(theta)  # channel-step per unit step
    dy_par = np.sin(theta)  # time-step per unit step

    # Create a 1D Gaussian sampling kernel
    radius = int(np.ceil(sigma_parallel * 3))
    steps = np.arange(-radius, radius + 1).astype(np.float32)
    weights_1d = np.exp(-0.5 * (steps / sigma_parallel) ** 2)
    weights_1d /= weights_1d.sum()

    # Sample along parallel direction
    accum = np.zeros_like(data, dtype=np.float32)
    weight_sum = np.zeros_like(data, dtype=np.float32) + 1e-12

    for step, w in zip(steps, weights_1d):
        # Small perpendicular blur by sigma_perp: mix with a slight
        # cross-direction smoothing via local Gaussian
        y_sample = yy + step * dy_par
        x_sample = xx + step * dx_par
        # Clip to valid region
        y0 = np.clip(np.floor(y_sample).astype(np.int32), 0, H - 1)
        y1 = np.clip(y0 + 1, 0, H - 1)
        x0 = np.clip(np.floor(x_sample).astype(np.int32), 0, W - 1)
        x1 = np.clip(x0 + 1, 0, W - 1)
        fy = y_sample - y0.astype(np.float32)
        fx = x_sample - x0.astype(np.float32)
        fy = fy[..., None] if fy.ndim < 2 else fy
        fx = fx[..., None] if fx.ndim < 2 else fx

        # Bilinear interpolation
        Ia = data[y0, x0]
        Ib = data[y0, x1]
        Ic = data[y1, x0]
        Id = data[y1, x1]
        top = Ia * (1 - fx) + Ib * fx
        bot = Ic * (1 - fx) + Id * fx
        sampled = top * (1 - fy) + bot * fy
        accum += w.astype(np.float32) * sampled
        weight_sum += w.astype(np.float32)

    smoothed_parallel = accum / weight_sum

    # Cross-direction (perpendicular) light smoothing via sigma_perpendicular
    if sigma_perpendicular > 0.1:
        # Perpendicular direction: rotate by 90 deg
        dx_perp = -np.sin(theta)
        dy_perp = np.cos(theta)
        rperp = int(np.ceil(sigma_perpendicular * 3))
        if rperp >= 1:
            steps_p = np.arange(-rperp, rperp + 1).astype(np.float32)
            w_p = np.exp(-0.5 * (steps_p / sigma_perpendicular) ** 2)
            w_p /= w_p.sum()
            accum2 = np.zeros_like(data)
            ws2 = np.zeros_like(data) + 1e-12
            for step, w in zip(steps_p, w_p):
                ys = np.clip((yy + step * dy_perp).astype(np.int32), 0, H - 1)
                xs = np.clip((xx + step * dx_perp).astype(np.int32), 0, W - 1)
                accum2 += w * data[ys, xs]
                ws2 += w
            smoothed_full = accum2 / ws2
            # Blend parallel and perpendicular (mostly parallel dominates)
            smoothed_parallel = 0.85 * smoothed_parallel + 0.15 * smoothed_full

    # Adaptive / fixed alpha blending
    if alpha_fixed is not None:
        # SP2P paper formula: smoothed = alpha * smoothed + (1-alpha) * original
        out = alpha_fixed * smoothed_parallel + (1.0 - alpha_fixed) * data
    elif coherence is not None:
        # Adaptive: smooth more where coherence is high (confident orientation)
        adaptive_alpha = np.clip(coherence, 0.0, 1.0)
        out = adaptive_alpha * smoothed_parallel + (1.0 - adaptive_alpha) * data
    else:
        out = smoothed_parallel

    return out


# ============================================================
# High-level API: Structure-guided filter (one-call)
# ============================================================
def structure_guided_filter(data,
                            tensor_window=(5, 5),
                            sigma_deriv=1.0,
                            sigma_parallel=3.0,
                            sigma_perpendicular=0.5,
                            alpha=0.8,
                            use_adaptive=False):
    """
    One-call structure-guided filtering for DAS waterfall images.

    Parameters:
        data: np.ndarray (H_time, W_channel)
        tensor_window: (T, X) samples for uniform window ST estimation
        sigma_deriv: pre-smoothing for gradient computation
        sigma_parallel: smoothing std (in samples) along wavefront strike
        sigma_perpendicular: smoothing std across wavefront (keep small!)
        alpha: fixed blending coefficient in [0,1]. alpha=1 -> full smooth,
               alpha=0 -> identity. SP2P paper uses alpha=0.8.
        use_adaptive: if True, ignores `alpha` and adapts via coherence

    Returns:
        enhanced: np.ndarray same shape as data
        info: dict of intermediate outputs for visualization
            {'theta': orientation map, 'coherence': coherence map,
             'lam1': larger eigenvalue, 'lam2': smaller eigenvalue}
    """
    # Step 1: Structure tensor
    J11, J12, J22 = compute_structure_tensor(
        data, sigma_deriv=sigma_deriv, window_size=tensor_window)

    # Step 2: Decompose
    theta, coherence, lam1, lam2 = decompose_structure_tensor(J11, J12, J22)

    # Step 3: Directional smooth
    alpha_used = None if use_adaptive else alpha
    enhanced = directional_smooth(
        data, theta,
        sigma_parallel=sigma_parallel,
        sigma_perpendicular=sigma_perpendicular,
        coherence=coherence,
        alpha_fixed=alpha_used)

    info = {'theta': theta, 'coherence': coherence,
            'lam1': lam1, 'lam2': lam2, 'J11': J11, 'J12': J12, 'J22': J22}
    return enhanced, info
