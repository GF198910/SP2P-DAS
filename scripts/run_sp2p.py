#!/usr/bin/env python3
"""
Run SP2P-DAS denoising on a single DAS segment.

Usage:
    python run_sp2p.py --data path/to/das.npy --output path/to/denoised.npy
    python run_sp2p.py --data path/to/das.npy --bidir  # enable bidirectional feedback
"""
import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import sp2p_das


def main():
    parser = argparse.ArgumentParser(description="SP2P-DAS Denoising")
    parser.add_argument("--data", required=True, help="Path to DAS data (.npy, shape: channels x time)")
    parser.add_argument("--output", default=None, help="Output path (.npy)")
    parser.add_argument("--bidir", action="store_true", help="Enable bidirectional structure feedback")
    parser.add_argument("--epochs", type=int, default=100, help="P2P training epochs")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable preprocessing (ablation)")
    args = parser.parse_args()

    data = np.load(args.data).astype(np.float32)
    print(f"Loaded data: shape={data.shape}, range=[{data.min():.1f}, {data.max():.1f}]")

    denoised = sp2p_das(
        data,
        bidirectional_guidance=args.bidir,
        p2p_max_epoch=args.epochs,
        verbose=True,
    )

    if args.output is None:
        base = os.path.splitext(args.data)[0]
        suffix = "_bidir" if args.bidir else ""
        args.output = f"{base}_sp2p{suffix}.npy"

    np.save(args.output, denoised)
    print(f"Saved: {args.output}  shape={denoised.shape}")


if __name__ == "__main__":
    main()
