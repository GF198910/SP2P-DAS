#!/usr/bin/env python3
"""
Run SP2P-DAS preprocessing ablation (8 configurations per paper).
Evaluates each of the 4 preprocessing components (BP, WS, ZS, BA) with toggles.
"""
import numpy as np
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import sp2p_das, Preprocessor
from src.fair_snr import compute_semblance_snr, compute_energy_snr


# 8 configurations matching the paper
ABLATION_CONFIGS = [
    # (name, bandpass, weak_suppress, zscore, border_atten)
    ("Full Preprocess",      True,  True,  True,  True),
    ("w/o Weak-suppress",   True,  False, True,  True),
    ("w/o Border atten.",   True,  True,  True,  False),
    ("w/o Z-score",         True,  True,  False, True),
    ("w/o Bandpass",        False, True,  True,  True),
    ("Bandpass only",       True,  False, False, False),
    ("Bandpass + Z-score",  True,  False, True,  False),
    ("No Preprocessing",    False, False, False, False),
]


def run_config(data, name, bp, ws, zs, ba):
    preprocessor = Preprocessor(
        do_bandpass=bp, lowcut=30.0, highcut=100.0, fs=1000,
        do_weak_suppress=ws, sigma_scale=2.0, weak_divisor=5.0,
        do_border_atten=ba, border_time=20, border_channel=50,
        do_zscore=zs,
    )
    denoised = sp2p_das(data, preprocessor=preprocessor, bidirectional_guidance=False, verbose=False)
    sembl_snr = compute_semblance_snr(denoised)
    energy_snr = compute_energy_snr(denoised)
    print(f"  {name:25s}  SemblSNR={sembl_snr:+.2f} dB  EnergySNR={energy_snr:+.2f} dB")
    return {"config": name, "sembl_snr_db": sembl_snr, "energy_snr_db": energy_snr}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to DAS segment (.npy)")
    parser.add_argument("--output", default="ablation_results.json")
    args = parser.parse_args()

    data = np.load(args.data).astype(np.float32)
    print(f"Ablation on {args.data} (shape={data.shape})")

    results = []
    for name, bp, ws, zs, ba in ABLATION_CONFIGS:
        r = run_config(data, name, bp, ws, zs, ba)
        results.append(r)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
