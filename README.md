# SP2P: Physics-Guided Non-Local Self-Similarity Denoising with Bidirectional Structure Feedback

A self-supervised two-stage denoising framework for **Distributed Acoustic Sensing (DAS)** data. SP2P requires **no external training data**, **no paired labels**, and **no clean target signals**.

## Overview

SP2P consists of two cascaded stages:

| Stage | Description |
|---|---|
| **Stage 1: DAS-adapted Pixel2Pixel** | Exploits non-local self-similarity in DAS waterfall images to construct zero-shot training pairs via block-wise patch matching and random sampling. A lightweight 6-layer fully-convolutional network learns signal structure from the noisy recording itself. |
| **Stage 2: Structure-Guided Enhancement** | Estimates local wavefront orientation via structure tensor decomposition and applies directional smoothing along seismic propagation directions, improving inter-channel coherence while preserving sharp event edges. |

A key innovation is the **bidirectional structure feedback**: the wavefront orientation estimated in Stage 2 is fed back to restrict the non-local patch search in Stage 1 to physically plausible directions (±30° tolerance), converting blind global matching into physics-guided directional search.

## Installation

```bash
git clone https://github.com/[YOUR_USERNAME]/SP2P.git
cd SP2P
pip install -r requirements.txt
```

### Dependencies
- PyTorch ≥ 1.13
- NumPy, SciPy, einops, tqdm
- ObsPy (for reading seismic data formats)
- Matplotlib (for visualization)

## Usage

### Basic denoising

```python
import numpy as np
from src import sp2p_das

# Load your DAS data (channels × time samples)
data = np.load("your_das_segment.npy")  # shape: (C, T)

# Run SP2P with default settings (matching the paper)
denoised = sp2p_das(data)

# Enable bidirectional structure feedback (paper's main configuration)
denoised_bidir = sp2p_das(data, bidirectional_guidance=True)
```

### Custom parameters

```python
from src import sp2p_das, Preprocessor

# Custom preprocessing (or disable entirely for ablation)
preprocessor = Preprocessor(
    do_bandpass=True, lowcut=30.0, highcut=100.0, fs=1000,
    do_weak_suppress=True, sigma_scale=2.0, weak_divisor=5.0,
    do_border_atten=True, border_time=20, border_channel=50,
    do_zscore=True,
)

denoised = sp2p_das(
    data,
    preprocessor=preprocessor,
    p2p_max_epoch=2000,       # training iterations
    p2p_num_neighbors=100,    # patch bank size
    bidirectional_guidance=True,
    angle_tolerance_deg=30.0, # directional search tolerance
)
```

## Data

The paper uses five labeled icequake events from two **Rutford Ice Stream, Antarctica** DAS segments, identical to the dataset used in ZSN2N-DAS [[1]](#references). Data are available from the original ZSN2N-DAS release.

## Reproducing the Paper Results

```bash
python scripts/run_sp2p.py --data data/seg1.npy --bidir --output results/seg1_denoised.npy
python scripts/run_ablation.py  # 8-configuration preprocessing ablation
python scripts/run_bidir_comparison.py  # bidirectional vs. blind search
```

## Evaluation Metrics

The repository includes implementations of the three metrics used in the paper:

- **Peak local semblance ratio** `S/(1-S)` — Marfurt semblance within event bounding boxes
- **Average semblance SNR** (dB) — mean `10log₁₀[S/(1-S)]` over event regions
- **Energy SNR** (dB) — `10log₁₀(E_signal / E_noise)` with manually delineated regions

## Repository Structure

```
SP2P/
├── src/
│   ├── sp2p.py              # Core SP2P-DAS pipeline (both stages + bidirectional feedback)
│   ├── preprocessing.py     # Modular preprocessing pipeline (4 switchable components)
│   ├── structure_filter.py  # Structure tensor computation + directional smoothing
│   ├── pixel2pixel.py       # Basic Pixel2Pixel patch matching utilities
│   ├── fair_snr.py          # Semblance and energy SNR metrics
│   └── bandpass.py          # Bandpass filtering helper
├── scripts/
│   ├── run_sp2p.py          # Entry point for denoising a single segment
│   ├── run_ablation.py      # Preprocessing ablation study
│   └── run_bidir_comparison.py
├── examples/
├── figures/                 # Paper figures
├── data/                    # Data directory (user-provided)
├── requirements.txt
└── README.md
```

## References

[1] Feng, Y., et al. "ZSN2N-DAS: Zero-Shot Noise2Noise Denoising for Distributed Acoustic Sensing Data." *Geophysical Journal International*, 2025.

## Citation

If you use this code, please cite:

```
[Full citation will be added upon publication]
```

## License

MIT License
