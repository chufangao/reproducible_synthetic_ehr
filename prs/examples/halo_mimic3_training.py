"""
Example: Training HALO on MIMIC-III for synthetic EHR generation.

This script demonstrates how to train the HALO model using PyHealth's
standard dataset and task patterns. HALO learns to generate synthetic
patient visit sequences via autoregressive transformer training.

Usage:
    python examples/halo_mimic3_training.py

Replace the ``root`` path below with the local path to your MIMIC-III
data directory before running.
"""

from pyhealth.datasets import MIMIC3Dataset, split_by_patient
from pyhealth.models.generators.halo import HALO
from pyhealth.tasks import halo_generation_mimic3_fn

# Step 1: Load MIMIC-III dataset
print("Loading MIMIC-III dataset...")
base_dataset = MIMIC3Dataset(
    root="/path/to/mimic3",
    tables=["diagnoses_icd"],
)
base_dataset.stats()

# Step 2: Set task for HALO generation
# halo_generation_mimic3_fn extracts diagnosis code sequences per patient.
# Each patient produces one sample with all their visits (admissions with
# at least one ICD-9 code). Patients with fewer than 2 qualifying visits
# are excluded.
print("Setting HALO generation task...")
sample_dataset = base_dataset.set_task(halo_generation_mimic3_fn)
print(f"Samples after task: {len(sample_dataset)}")

# Step 3: Split dataset by patient (no patient appears in more than one split)
print("Splitting dataset...")
train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, [0.8, 0.1, 0.1]
)
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Step 4: Initialize HALO model
# The model derives vocabulary size automatically from the dataset's
# NestedSequenceProcessor. No manual vocabulary setup is needed.
print("Initializing HALO model...")
model = HALO(
    dataset=sample_dataset,
    embed_dim=768,
    n_heads=12,
    n_layers=12,
    n_ctx=48,
    batch_size=48,
    epochs=50,
    pos_loss_weight=None,
    lr=1e-4,
    save_dir="./save/",
)

# Step 5: Train using HALO's custom training loop
# HALO does not use the PyHealth Trainer; it has its own loop that
# validates after every epoch and saves the best checkpoint to save_dir.
print("Starting training...")
model.train_model(train_dataset, val_dataset)

print("Training complete. Best checkpoint saved to ./save/halo_model")
