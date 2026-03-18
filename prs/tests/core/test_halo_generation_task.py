"""Tests for HaloGenerationMIMIC3 task function."""

import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks.base_task import BaseTask
from pyhealth.tasks.halo_generation import (
    HaloGenerationMIMIC3,
    HaloGenerationMIMIC4,
    halo_generation_mimic3_fn,
    halo_generation_mimic4_fn,
)


def _make_patient_df(admissions, diagnoses):
    """Build a minimal Polars DataFrame for a Patient.

    Args:
        admissions: list of dicts with keys: hadm_id, admittime, dischtime
        diagnoses: list of dicts with keys: hadm_id, icd9_code
            Use icd9_code=None to simulate a null/missing code.

    Returns:
        pl.DataFrame suitable for Patient(patient_id, df).
    """
    rows = []

    for adm in admissions:
        rows.append(
            {
                "event_type": "admissions",
                "timestamp": adm["admittime"],
                "admissions/hadm_id": str(adm["hadm_id"]),
                "admissions/admittime": adm["admittime"],
                "admissions/dischtime": adm.get("dischtime", adm["admittime"]),
                "diagnoses_icd/hadm_id": None,
                "diagnoses_icd/icd9_code": None,
            }
        )

    for diag in diagnoses:
        rows.append(
            {
                "event_type": "diagnoses_icd",
                "timestamp": diag.get("timestamp", datetime(2100, 1, 1)),
                "admissions/hadm_id": None,
                "admissions/admittime": None,
                "admissions/dischtime": None,
                "diagnoses_icd/hadm_id": str(diag["hadm_id"]),
                "diagnoses_icd/icd9_code": diag.get("icd9_code"),
            }
        )

    schema = {
        "event_type": pl.Utf8,
        "timestamp": pl.Datetime,
        "admissions/hadm_id": pl.Utf8,
        "admissions/admittime": pl.Datetime,
        "admissions/dischtime": pl.Datetime,
        "diagnoses_icd/hadm_id": pl.Utf8,
        "diagnoses_icd/icd9_code": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


class TestHaloGenerationMIMIC3TaskSchema(unittest.TestCase):
    """Tests for task class attributes (schema, naming)."""

    def test_task_name(self):
        self.assertEqual(HaloGenerationMIMIC3.task_name, "HaloGenerationMIMIC3")

    def test_input_schema(self):
        self.assertIn("visits", HaloGenerationMIMIC3.input_schema)
        self.assertEqual(HaloGenerationMIMIC3.input_schema["visits"], "nested_sequence")

    def test_output_schema_is_empty(self):
        self.assertEqual(HaloGenerationMIMIC3.output_schema, {})

    def test_inherits_base_task(self):
        task = HaloGenerationMIMIC3()
        self.assertIsInstance(task, BaseTask)

    def test_convenience_instance_is_base_task(self):
        self.assertIsInstance(halo_generation_mimic3_fn, BaseTask)
        self.assertIsInstance(halo_generation_mimic3_fn, HaloGenerationMIMIC3)

    def test_convenience_instance_is_callable(self):
        self.assertTrue(callable(halo_generation_mimic3_fn))


class TestHaloGenerationMIMIC3Call(unittest.TestCase):
    """Unit tests for HaloGenerationMIMIC3.__call__ using mock Patient objects."""

    def setUp(self):
        self.task = HaloGenerationMIMIC3()

    def _make_patient(self, patient_id, admissions, diagnoses):
        df = _make_patient_df(admissions, diagnoses)
        return Patient(patient_id=patient_id, data_source=df)

    def test_valid_patient_two_visits_returns_one_sample(self):
        """Patient with 2 admissions each having diagnosis codes returns 1 sample."""
        patient = self._make_patient(
            patient_id="P001",
            admissions=[
                {"hadm_id": "100", "admittime": datetime(2100, 1, 1)},
                {"hadm_id": "200", "admittime": datetime(2101, 1, 1)},
            ],
            diagnoses=[
                {"hadm_id": "100", "icd9_code": "D001", "timestamp": datetime(2100, 1, 2)},
                {"hadm_id": "100", "icd9_code": "D002", "timestamp": datetime(2100, 1, 2)},
                {"hadm_id": "200", "icd9_code": "D003", "timestamp": datetime(2101, 1, 2)},
            ],
        )
        result = self.task(patient)
        self.assertEqual(len(result), 1)

    def test_valid_patient_sample_structure(self):
        """Sample must contain patient_id and visits keys."""
        patient = self._make_patient(
            patient_id="P001",
            admissions=[
                {"hadm_id": "100", "admittime": datetime(2100, 1, 1)},
                {"hadm_id": "200", "admittime": datetime(2101, 1, 1)},
            ],
            diagnoses=[
                {"hadm_id": "100", "icd9_code": "D001", "timestamp": datetime(2100, 1, 2)},
                {"hadm_id": "200", "icd9_code": "D003", "timestamp": datetime(2101, 1, 2)},
            ],
        )
        result = self.task(patient)
        sample = result[0]
        self.assertIn("patient_id", sample)
        self.assertIn("visits", sample)

    def test_valid_patient_patient_id_correct(self):
        """Sample patient_id must match the patient's ID."""
        patient = self._make_patient(
            patient_id="P42",
            admissions=[
                {"hadm_id": "100", "admittime": datetime(2100, 1, 1)},
                {"hadm_id": "200", "admittime": datetime(2101, 1, 1)},
            ],
            diagnoses=[
                {"hadm_id": "100", "icd9_code": "D001", "timestamp": datetime(2100, 1, 2)},
                {"hadm_id": "200", "icd9_code": "D003", "timestamp": datetime(2101, 1, 2)},
            ],
        )
        result = self.task(patient)
        self.assertEqual(result[0]["patient_id"], "P42")

    def test_valid_patient_visits_are_nested_list(self):
        """visits must be a list of lists of strings."""
        patient = self._make_patient(
            patient_id="P001",
            admissions=[
                {"hadm_id": "100", "admittime": datetime(2100, 1, 1)},
                {"hadm_id": "200", "admittime": datetime(2101, 1, 1)},
            ],
            diagnoses=[
                {"hadm_id": "100", "icd9_code": "D001", "timestamp": datetime(2100, 1, 2)},
                {"hadm_id": "100", "icd9_code": "D002", "timestamp": datetime(2100, 1, 2)},
                {"hadm_id": "200", "icd9_code": "D003", "timestamp": datetime(2101, 1, 2)},
            ],
        )
        result = self.task(patient)
        visits = result[0]["visits"]
        self.assertIsInstance(visits, list)
        for visit in visits:
            self.assertIsInstance(visit, list)
            for code in visit:
                self.assertIsInstance(code, str)

    def test_valid_patient_codes_are_correct(self):
        """Codes in visits must match the ICD codes from the input data."""
        patient = self._make_patient(
            patient_id="P001",
            admissions=[
                {"hadm_id": "100", "admittime": datetime(2100, 1, 1)},
                {"hadm_id": "200", "admittime": datetime(2101, 1, 1)},
            ],
            diagnoses=[
                {"hadm_id": "100", "icd9_code": "D001", "timestamp": datetime(2100, 1, 2)},
                {"hadm_id": "100", "icd9_code": "D002", "timestamp": datetime(2100, 1, 2)},
                {"hadm_id": "200", "icd9_code": "D003", "timestamp": datetime(2101, 1, 2)},
            ],
        )
        result = self.task(patient)
        visits = result[0]["visits"]
        self.assertEqual(len(visits), 2)
        self.assertIn("D001", visits[0])
        self.assertIn("D002", visits[0])
        self.assertIn("D003", visits[1])

    def test_patient_with_one_admission_returns_empty(self):
        """Patients with only 1 admission are excluded."""
        patient = self._make_patient(
            patient_id="P001",
            admissions=[
                {"hadm_id": "100", "admittime": datetime(2100, 1, 1)},
            ],
            diagnoses=[
                {"hadm_id": "100", "icd9_code": "D001", "timestamp": datetime(2100, 1, 2)},
            ],
        )
        result = self.task(patient)
        self.assertEqual(result, [])

    def test_patient_with_zero_admissions_returns_empty(self):
        """Patients with no admissions return empty list."""
        patient = self._make_patient(
            patient_id="P001",
            admissions=[],
            diagnoses=[],
        )
        result = self.task(patient)
        self.assertEqual(result, [])

    def test_patient_only_one_visit_has_codes_returns_empty(self):
        """Patient with 2 admissions but only 1 with diagnosis codes returns empty."""
        patient = self._make_patient(
            patient_id="P001",
            admissions=[
                {"hadm_id": "100", "admittime": datetime(2100, 1, 1)},
                {"hadm_id": "200", "admittime": datetime(2101, 1, 1)},
            ],
            diagnoses=[
                # Only hadm_id 100 has codes; 200 has none
                {"hadm_id": "100", "icd9_code": "D001", "timestamp": datetime(2100, 1, 2)},
            ],
        )
        result = self.task(patient)
        self.assertEqual(result, [])

    def test_patient_with_null_codes_skipped(self):
        """Null diagnosis codes must be dropped; if only nulls in a visit, skip it."""
        patient = self._make_patient(
            patient_id="P001",
            admissions=[
                {"hadm_id": "100", "admittime": datetime(2100, 1, 1)},
                {"hadm_id": "200", "admittime": datetime(2101, 1, 1)},
                {"hadm_id": "300", "admittime": datetime(2102, 1, 1)},
            ],
            diagnoses=[
                # hadm 100: only a null code → should be skipped
                {"hadm_id": "100", "icd9_code": None, "timestamp": datetime(2100, 1, 2)},
                # hadm 200: valid code
                {"hadm_id": "200", "icd9_code": "D003", "timestamp": datetime(2101, 1, 2)},
                # hadm 300: valid code
                {"hadm_id": "300", "icd9_code": "D004", "timestamp": datetime(2102, 1, 2)},
            ],
        )
        result = self.task(patient)
        # hadm 100 skipped → only 2 visits remain → valid patient
        self.assertEqual(len(result), 1)
        visits = result[0]["visits"]
        self.assertEqual(len(visits), 2)
        self.assertIn("D003", visits[0])
        self.assertIn("D004", visits[1])

    def test_three_valid_visits_all_included(self):
        """All visits with codes are included in the nested list."""
        patient = self._make_patient(
            patient_id="P001",
            admissions=[
                {"hadm_id": "100", "admittime": datetime(2100, 1, 1)},
                {"hadm_id": "200", "admittime": datetime(2101, 1, 1)},
                {"hadm_id": "300", "admittime": datetime(2102, 1, 1)},
            ],
            diagnoses=[
                {"hadm_id": "100", "icd9_code": "A", "timestamp": datetime(2100, 1, 2)},
                {"hadm_id": "200", "icd9_code": "B", "timestamp": datetime(2101, 1, 2)},
                {"hadm_id": "300", "icd9_code": "C", "timestamp": datetime(2102, 1, 2)},
            ],
        )
        result = self.task(patient)
        self.assertEqual(len(result), 1)
        visits = result[0]["visits"]
        self.assertEqual(len(visits), 3)


class TestHaloGenerationMIMIC4TaskSchema(unittest.TestCase):
    """Tests for HaloGenerationMIMIC4 class attributes."""

    def test_task_name(self):
        self.assertEqual(HaloGenerationMIMIC4.task_name, "HaloGenerationMIMIC4")

    def test_inherits_mimic3(self):
        task = HaloGenerationMIMIC4()
        self.assertIsInstance(task, HaloGenerationMIMIC3)

    def test_convenience_instance(self):
        self.assertIsInstance(halo_generation_mimic4_fn, HaloGenerationMIMIC4)
        self.assertIsInstance(halo_generation_mimic4_fn, BaseTask)


class TestHaloGenerationMIMIC3Integration(unittest.TestCase):
    """Integration tests using the MIMIC-III demo dataset."""

    @classmethod
    def setUpClass(cls):
        try:
            import inspect

            from pyhealth.datasets import MIMIC3Dataset

            if not inspect.isclass(MIMIC3Dataset):
                raise ImportError(
                    "pyhealth.datasets.MIMIC3Dataset is not a real class "
                    "(stub injected by another test module); skipping integration tests."
                )

            demo_path = str(
                Path(__file__).parent.parent.parent
                / "test-resources"
                / "core"
                / "mimic3demo"
            )
            cls.dataset = MIMIC3Dataset(
                root=demo_path,
                tables=["diagnoses_icd"],
            )
            cls.task = HaloGenerationMIMIC3()
            cls.sample_dataset = cls.dataset.set_task(cls.task)
            cls.skip_integration = False
        except (FileNotFoundError, OSError, ImportError, ValueError) as e:
            cls.skip_integration = True
            cls.skip_reason = str(e)

    def setUp(self):
        if self.skip_integration:
            self.skipTest(f"Integration test skipped: {self.skip_reason}")

    def test_set_task_returns_dataset(self):
        self.assertIsNotNone(self.sample_dataset)

    def test_samples_generated(self):
        """Should produce at least one sample from the demo dataset."""
        self.assertGreater(len(self.sample_dataset), 0)

    def test_sample_has_required_keys(self):
        """Every sample must have patient_id and visits."""
        for sample in self.sample_dataset:
            self.assertIn("patient_id", sample)
            self.assertIn("visits", sample)

    def test_visits_are_nested_lists(self):
        """visits must be a list of lists of strings."""
        for sample in self.sample_dataset:
            visits = sample["visits"]
            self.assertIsInstance(visits, list)
            self.assertGreaterEqual(len(visits), 2)
            for visit in visits:
                self.assertIsInstance(visit, list)
                self.assertGreater(len(visit), 0)
                for code in visit:
                    self.assertIsInstance(code, str)

    def test_no_patients_with_single_visit(self):
        """No sample should come from a patient with only one valid visit."""
        for sample in self.sample_dataset:
            self.assertGreaterEqual(
                len(sample["visits"]),
                2,
                f"Patient {sample['patient_id']} has fewer than 2 visits",
            )


if __name__ == "__main__":
    unittest.main()
