"""End-to-end integration tests for the HALO synthetic EHR generation pipeline.

Category A tests use InMemorySampleDataset with synthetic data — no external
data required and must always pass.

Category B tests require actual MIMIC-III data and are skipped gracefully when
the data is unavailable.

The bootstrap pattern (loading HALO and InMemorySampleDataset via importlib
while stubbing out heavy optional dependencies) mirrors the approach used in
tests/core/test_halo_model.py.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Bootstrap: load HALO, BaseModel, and InMemorySampleDataset without
# triggering pyhealth.models.__init__ (requires einops, litdata, etc.) or
# pyhealth.datasets.__init__ (requires litdata, pyarrow, pandas, dask, ...).
# ---------------------------------------------------------------------------

def _bootstrap():
    """Load HALO, BaseModel, and InMemorySampleDataset via importlib.

    Returns:
        (BaseModel, HALO, InMemorySampleDataset)
    """
    import pyhealth  # noqa: F401  — top-level __init__ has no heavy deps

    # Stub pyhealth.datasets so that base_model.py's
    # "from ..datasets import SampleDataset" resolves cleanly.
    if "pyhealth.datasets" not in sys.modules:
        ds_stub = MagicMock()
        class _FakeSampleDataset:  # noqa: N801
            pass
        ds_stub.SampleDataset = _FakeSampleDataset
        sys.modules["pyhealth.datasets"] = ds_stub

    # Stub pyhealth.models so we can control loading without the real __init__.
    if "pyhealth.models" not in sys.modules or isinstance(
        sys.modules["pyhealth.models"], MagicMock
    ):
        models_stub = MagicMock()
        sys.modules["pyhealth.models"] = models_stub
    else:
        models_stub = sys.modules["pyhealth.models"]

    # Processors are safe to import normally.
    from pyhealth.processors import PROCESSOR_REGISTRY  # noqa: F401

    def _load_file(mod_name, filepath):
        spec = importlib.util.spec_from_file_location(mod_name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base = os.path.join(root, "pyhealth", "models")
    res = os.path.join(base, "generators", "halo_resources")

    # Load base_model and expose via stub.
    bm_mod = _load_file("pyhealth.models.base_model", os.path.join(base, "base_model.py"))
    BaseModel = bm_mod.BaseModel
    models_stub.BaseModel = BaseModel

    # Load HALO sub-dependencies.
    _load_file(
        "pyhealth.models.generators.halo_resources.halo_config",
        os.path.join(res, "halo_config.py"),
    )
    _load_file(
        "pyhealth.models.generators.halo_resources.halo_model",
        os.path.join(res, "halo_model.py"),
    )

    gen_stub = MagicMock()
    sys.modules.setdefault("pyhealth.models.generators", gen_stub)

    halo_mod = _load_file(
        "pyhealth.models.generators.halo",
        os.path.join(base, "generators", "halo.py"),
    )
    HALO = halo_mod.HALO

    # Stub litdata so sample_dataset.py can be loaded without the full package.
    # sample_dataset.py imports litdata.StreamingDataset and
    # litdata.utilities.train_test_split.deepcopy_dataset.
    if "litdata" not in sys.modules:
        litdata_pkg = MagicMock()
        litdata_pkg.StreamingDataset = type(
            "StreamingDataset", (), {"__init__": lambda self, *a, **kw: None}
        )
        litdata_utilities = MagicMock()
        litdata_utilities_train_test = MagicMock()
        litdata_utilities_train_test.deepcopy_dataset = lambda x: x
        litdata_utilities.train_test_split = litdata_utilities_train_test
        litdata_pkg.utilities = litdata_utilities
        sys.modules["litdata"] = litdata_pkg
        sys.modules["litdata.utilities"] = litdata_utilities
        sys.modules["litdata.utilities.train_test_split"] = litdata_utilities_train_test

    # Load sample_dataset.py directly (bypasses datasets/__init__.py).
    ds_file_mod = _load_file(
        "pyhealth.datasets.sample_dataset",
        os.path.join(root, "pyhealth", "datasets", "sample_dataset.py"),
    )
    InMemorySampleDataset = ds_file_mod.InMemorySampleDataset

    return BaseModel, HALO, InMemorySampleDataset


BaseModel, HALO, InMemorySampleDataset = _bootstrap()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

_SMALL_SAMPLES = [
    {"patient_id": "p1", "visits": [["A", "B"], ["C"]]},
    {"patient_id": "p2", "visits": [["A"], ["B", "C"]]},
    {"patient_id": "p3", "visits": [["D", "E"], ["F", "G"], ["H"]]},
    {"patient_id": "p4", "visits": [["A", "B"], ["C", "D"]]},
    {"patient_id": "p5", "visits": [["B", "C"], ["D", "E"]]},
]

_SMALL_MODEL_KWARGS = dict(
    embed_dim=32,
    n_heads=2,
    n_layers=1,
    n_ctx=8,
    batch_size=4,
    epochs=1,
)


def _make_dataset(samples=None):
    if samples is None:
        samples = _SMALL_SAMPLES
    return InMemorySampleDataset(
        samples=samples,
        input_schema={"visits": "nested_sequence"},
        output_schema={},
    )


# ---------------------------------------------------------------------------
# Category A: In-Memory Integration Tests (must always pass)
# ---------------------------------------------------------------------------


class TestHALOFullPipelineForward(unittest.TestCase):
    """Full pipeline: build dataset -> create HALO -> forward pass."""

    def setUp(self):
        self.dataset = _make_dataset()
        self.tmpdir = tempfile.mkdtemp()
        self.model = HALO(self.dataset, save_dir=self.tmpdir, **_SMALL_MODEL_KWARGS)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_pipeline_forward(self):
        """Forward pass on a batch from the dataset returns loss and predictions."""
        sample = self.dataset[0]
        visits = sample["visits"].unsqueeze(0)  # add batch dimension
        output = self.model(visits=visits)
        self.assertIsInstance(output, dict)
        self.assertIn("loss", output)
        self.assertIn("predictions", output)
        self.assertIsInstance(output["loss"], torch.Tensor)
        self.assertEqual(output["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(output["loss"]))


class TestHALOTrainModelOneEpoch(unittest.TestCase):
    """train_model completes one epoch without error."""

    def test_train_model_runs_one_epoch(self):
        dataset = _make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            model = HALO(dataset, save_dir=tmpdir, **_SMALL_MODEL_KWARGS)
            # Should complete without raising any exception.
            try:
                model.train_model(dataset, val_dataset=None)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"train_model raised an unexpected exception: {exc}")


class TestHALOCheckpointSavedAfterTraining(unittest.TestCase):
    """Checkpoint file exists after train_model with a validation dataset."""

    def test_checkpoint_saved_after_training(self):
        dataset = _make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            model = HALO(dataset, save_dir=tmpdir, **_SMALL_MODEL_KWARGS)
            # Checkpoint is only written when val_dataset is provided and
            # the validation loss improves.
            model.train_model(dataset, val_dataset=dataset)
            checkpoint_path = os.path.join(tmpdir, "halo_model")
            self.assertTrue(
                os.path.exists(checkpoint_path),
                f"Expected checkpoint at {checkpoint_path}",
            )


class TestHALOSynthesizeReturnsCorrectCount(unittest.TestCase):
    """synthesize_dataset(num_samples=5) returns exactly 5 dicts."""

    def setUp(self):
        self.dataset = _make_dataset()
        self.tmpdir = tempfile.mkdtemp()
        self.model = HALO(self.dataset, save_dir=self.tmpdir, **_SMALL_MODEL_KWARGS)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_synthesize_returns_correct_count(self):
        result = self.model.synthesize_dataset(num_samples=5)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)


class TestHALOSynthesizeOutputStructure(unittest.TestCase):
    """Each synthesized dict has patient_id and visits keys."""

    def setUp(self):
        self.dataset = _make_dataset()
        self.tmpdir = tempfile.mkdtemp()
        self.model = HALO(self.dataset, save_dir=self.tmpdir, **_SMALL_MODEL_KWARGS)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_synthesize_output_structure(self):
        result = self.model.synthesize_dataset(num_samples=3)
        for i, item in enumerate(result):
            self.assertIsInstance(item, dict, f"Item {i} is not a dict")
            self.assertIn("patient_id", item, f"Item {i} missing 'patient_id'")
            self.assertIn("visits", item, f"Item {i} missing 'visits'")
            self.assertIsInstance(item["visits"], list, f"visits in item {i} is not a list")
            for j, visit in enumerate(item["visits"]):
                self.assertIsInstance(
                    visit, list, f"visit {j} in item {i} is not a list"
                )


class TestHALOVariableVisitCounts(unittest.TestCase):
    """DataLoader does not crash when patients have different numbers of visits."""

    def test_pipeline_with_variable_visit_counts(self):
        # Patients with 2, 3, and 5 visits respectively.
        samples = [
            {"patient_id": "p1", "visits": [["A", "B"], ["C"]]},
            {"patient_id": "p2", "visits": [["A"], ["B", "C"], ["D"]]},
            {"patient_id": "p3", "visits": [["E"], ["F"], ["G"], ["H"], ["I"]]},
        ]
        dataset = _make_dataset(samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            model = HALO(dataset, save_dir=tmpdir, **_SMALL_MODEL_KWARGS)
            # If _halo_collate_fn is missing or broken, DataLoader will raise
            # RuntimeError due to mismatched visit-dimension sizes.
            try:
                model.train_model(dataset, val_dataset=None)
            except RuntimeError as exc:
                self.fail(
                    f"train_model raised RuntimeError with variable visit counts: {exc}"
                )


class TestHALOIsBaseModelInstance(unittest.TestCase):
    """HALO model is an instance of BaseModel."""

    def test_model_is_basemodel_instance(self):
        dataset = _make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            model = HALO(dataset, save_dir=tmpdir, **_SMALL_MODEL_KWARGS)
        self.assertIsInstance(model, BaseModel)


class TestHALOFeatureKeys(unittest.TestCase):
    """model.feature_keys equals ['visits']."""

    def test_feature_keys(self):
        dataset = _make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            model = HALO(dataset, save_dir=tmpdir, **_SMALL_MODEL_KWARGS)
        self.assertEqual(model.feature_keys, ["visits"])


# ---------------------------------------------------------------------------
# Category B: MIMIC-III Integration Tests (skipped if data unavailable)
# ---------------------------------------------------------------------------

_MIMIC3_PATH = os.environ.get(
    "PYHEALTH_MIMIC3_PATH",
    "/srv/local/data/physionet.org/files/mimiciii/1.4",
)


class TestHALOMIMIC3Integration(unittest.TestCase):
    """End-to-end pipeline test with actual MIMIC-III data.

    Skipped automatically when MIMIC-III is not present on this machine.
    """

    @classmethod
    def setUpClass(cls):
        cls.skip_integration = False
        cls.skip_reason = ""
        try:
            # Remove the bootstrap stub for pyhealth.datasets so we can attempt
            # a real import (which will raise ImportError if litdata is absent).
            _saved_stub = sys.modules.pop("pyhealth.datasets", None)
            try:
                import importlib as _il
                _il.invalidate_caches()
                from pyhealth.datasets import MIMIC3Dataset as _MIMIC3Dataset
                from pyhealth.tasks.halo_generation import HaloGenerationMIMIC3
            except (ImportError, ModuleNotFoundError) as exc:
                # Restore stub so the rest of the test session is unaffected.
                if _saved_stub is not None:
                    sys.modules["pyhealth.datasets"] = _saved_stub
                raise ImportError(str(exc)) from exc
            # Restore whatever was there (real module or stub).
            # If the import succeeded, sys.modules["pyhealth.datasets"] is now the
            # real module — keep it.

            cls.dataset = _MIMIC3Dataset(
                root=_MIMIC3_PATH,
                tables=["diagnoses_icd"],
            )
            task = HaloGenerationMIMIC3()
            cls.sample_dataset = cls.dataset.set_task(task)
        except (FileNotFoundError, OSError, ImportError, ValueError) as exc:
            cls.skip_integration = True
            cls.skip_reason = str(exc)

    def setUp(self):
        if self.skip_integration:
            self.skipTest(f"MIMIC-III integration test skipped: {self.skip_reason}")

    def test_mimic3_set_task_returns_nonempty_dataset(self):
        """set_task produces at least one sample from MIMIC-III."""
        self.assertGreater(len(self.sample_dataset), 0)

    def test_mimic3_sample_keys(self):
        """Every sample must contain patient_id and visits keys."""
        for sample in self.sample_dataset:
            self.assertIn("patient_id", sample)
            self.assertIn("visits", sample)

    def test_mimic3_visits_are_nested_lists_of_strings(self):
        """visits must be a list of lists of strings with at least 2 visits."""
        for sample in self.sample_dataset:
            visits = sample["visits"]
            self.assertIsInstance(visits, list)
            self.assertGreaterEqual(len(visits), 2)
            for visit in visits:
                self.assertIsInstance(visit, list)
                self.assertGreater(len(visit), 0)
                for code in visit:
                    self.assertIsInstance(code, str)

    def test_mimic3_full_pipeline_train_and_synthesize(self):
        """Train one epoch on MIMIC-III data and synthesize a small batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = HALO(
                self.sample_dataset,
                embed_dim=32,
                n_heads=2,
                n_layers=1,
                n_ctx=8,
                batch_size=16,
                epochs=1,
                save_dir=tmpdir,
            )
            model.train_model(self.sample_dataset, val_dataset=None)
            synthetic = model.synthesize_dataset(num_samples=10)
            self.assertEqual(len(synthetic), 10)
            for item in synthetic:
                self.assertIn("patient_id", item)
                self.assertIn("visits", item)


if __name__ == "__main__":
    unittest.main()
