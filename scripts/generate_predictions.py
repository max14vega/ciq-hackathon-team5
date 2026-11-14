
"""
Project Prometheus - Synthetic Test Prediction Generator

Creates 18 synthetic TIFF files matching the E6 experiment format:
    2 segments x 3 checkpoints x 3 depth windows

These files are for testing the analysis pipeline ONLY.
They are NOT real Vesuvius model predictions.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile


CHECKPOINTS = ["ck010k", "ck040k", "ck075k"]
DEPTH_WINDOWS = ["L04-20", "L06-22", "L08-24"]
SEGMENTS = ["w024", "w043"]


def generate_predictions(output_root, overwrite=False):
    output_root = Path(output_root)

    # Reproducible random numbers
    rng = np.random.default_rng(42)

    height = 128
    width = 192

    y, x = np.mgrid[:height, :width]

    # Artificial ink-like pattern
    ink = (
        np.sin(x / 11) * np.cos(y / 17) > 0.45
    ).astype(np.float32)

    created = 0
    skipped = 0

    for segment in SEGMENTS:
        segment_dir = output_root / segment
        segment_dir.mkdir(parents=True, exist_ok=True)

        for ckpt_index, checkpoint in enumerate(CHECKPOINTS):
            for depth_index, depth in enumerate(DEPTH_WINDOWS):

                filename = (
                    f"{segment}__{checkpoint}__{depth}.tif"
                )
                output_path = segment_dir / filename

                # Never overwrite existing files by default
                if output_path.exists() and not overwrite:
                    print(f"SKIPPED: {output_path}")
                    skipped += 1
                    continue

                # Simulate changes between model checkpoints
                checkpoint_shift = (ckpt_index - 1) * 0.055

                # Simulate changes between depth windows
                depth_shift = (depth_index - 1) * 0.035

                noise = rng.normal(
                    loc=0,
                    scale=0.06,
                    size=ink.shape
                )

                # Simulated ink scores
                prediction = (
                    0.35
                    + 0.40 * ink
                    + checkpoint_shift
                    + depth_shift
                    + noise
                )

                prediction = np.clip(
                    prediction, 0.0, 1.0
                )

                # Simulate area outside the papyrus
                prediction[:8, :] = 0.0

                # Match real Fuzzball TIFF format: uint8
                prediction_uint8 = np.rint(
                    prediction * 255
                ).astype(np.uint8)

                tifffile.imwrite(
                    output_path,
                    prediction_uint8,
                    photometric="minisblack"
                )

                print(f"CREATED: {output_path}")
                created += 1

    print("\n--- Generation Complete ---")
    print(f"Files created: {created}")
    print(f"Files skipped: {skipped}")
    print(f"Output directory: {output_root}")
    print("Expected total: 18 TIFF files")
    print("\nThese are SYNTHETIC test predictions.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic E6 prediction TIFFs."
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/synthetic_E6"),
        help="Directory for synthetic TIFF files"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing synthetic TIFF files"
    )

    args = parser.parse_args()

    # Safety: prevent accidental writes into real experiment data.
    output_path = args.output_root.resolve()
    if "synthetic" not in str(output_path).lower():
        parser.error(
            "Output path must contain 'synthetic' "
            "to protect real prediction files."
        )

    generate_predictions(
        output_root=args.output_root,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
