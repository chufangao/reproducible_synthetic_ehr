#!/usr/bin/env python3
"""
Example: Generate synthetic MIMIC-III patients using a trained HALO checkpoint.

Loads MIMIC3Dataset with the halo_generation_mimic3_fn task (identical to
training) so that the vocabulary is reconstructed, then loads the saved
checkpoint and calls model.synthesize_dataset(). Output is saved as JSON.

Usage:
    python examples/generate_synthetic_mimic3_halo.py
    python examples/generate_synthetic_mimic3_halo.py --save_dir ./my_save/ --num_samples 500
"""

import argparse
import json
import os

import torch

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.models.generators.halo import HALO
from pyhealth.tasks import halo_generation_mimic3_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic MIMIC-III patients with HALO"
    )
    parser.add_argument(
        "--mimic3_root",
        default="/path/to/mimic3",
        help="Root directory of MIMIC-III data (default: /path/to/mimic3)",
    )
    parser.add_argument(
        "--save_dir",
        default="./save/",
        help="Directory containing the trained halo_model checkpoint (default: ./save/)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of synthetic patients to generate (default: 1000)",
    )
    parser.add_argument(
        "--output",
        default="synthetic_patients.json",
        help="Output JSON file path (default: synthetic_patients.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # STEP 1: Load MIMIC-III dataset
    # The dataset must use the same tables and code_mapping as training
    # so that the vocabulary is identical.
    # ------------------------------------------------------------------
    print("Loading MIMIC-III dataset...")
    base_dataset = MIMIC3Dataset(
        root=args.mimic3_root,
        tables=["diagnoses_icd"],  # If you trained with different tables=, update this to match.
        code_mapping={},
        dev=False,
        refresh_cache=False,
    )
    print(f"  Loaded {len(base_dataset.patients)} patients")

    # ------------------------------------------------------------------
    # STEP 2: Apply the HALO generation task
    # set_task builds the vocabulary via NestedSequenceProcessor — must
    # match the task used during training exactly.
    # ------------------------------------------------------------------
    print("Applying HALO generation task...")
    sample_dataset = base_dataset.set_task(halo_generation_mimic3_fn)
    print(f"  {len(sample_dataset)} samples after task filtering")

    # ------------------------------------------------------------------
    # STEP 3: Instantiate HALO with the same hyperparameters as training
    # The model constructor uses the dataset to determine vocab sizes;
    # the weights are loaded from the checkpoint immediately after.
    # ------------------------------------------------------------------
    print("Initializing HALO model...")
    model = HALO(
        dataset=sample_dataset,
        embed_dim=768,
        n_heads=12,
        n_layers=12,
        n_ctx=48,
        batch_size=48,
        epochs=50,  # unused during generation; must match training for checkpoint compatibility
        pos_loss_weight=None,
        lr=1e-4,
        save_dir=args.save_dir,
    )

    # ------------------------------------------------------------------
    # STEP 4: Load trained checkpoint
    # The training loop saves to save_dir/halo_model with keys
    # "model" (halo_model state dict) and "optimizer".
    # ------------------------------------------------------------------
    checkpoint_path = os.path.join(args.save_dir, "halo_model")
    print(f"Loading checkpoint from {checkpoint_path} ...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Train the model first with examples/halo_mimic3_training.py."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.halo_model.load_state_dict(checkpoint["model"])
    print("  Checkpoint loaded successfully")

    # ------------------------------------------------------------------
    # STEP 5: Generate synthetic patients
    # synthesize_dataset returns List[Dict] where each dict has:
    #   "patient_id": "synthetic_N"
    #   "visits": [[code, ...], ...]
    # ------------------------------------------------------------------
    print(f"Generating {args.num_samples} synthetic patients...")
    synthetic_data = model.synthesize_dataset(
        num_samples=args.num_samples,
        random_sampling=True,
    )

    # ------------------------------------------------------------------
    # STEP 6: Save output as JSON
    # ------------------------------------------------------------------
    print(f"Saving output to {args.output} ...")
    with open(args.output, "w") as f:
        json.dump(synthetic_data, f, indent=2)

    # ------------------------------------------------------------------
    # STEP 7: Print summary statistics
    # ------------------------------------------------------------------
    total_patients = len(synthetic_data)
    total_visits = sum(len(p["visits"]) for p in synthetic_data)
    avg_visits = total_visits / total_patients if total_patients > 0 else 0.0

    print("\n--- Generation Summary ---")
    print(f"  Patients generated : {total_patients}")
    print(f"  Total visits       : {total_visits}")
    print(f"  Avg visits/patient : {avg_visits:.2f}")
    print(f"  Output saved to    : {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
