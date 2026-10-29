import ast
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("cuda_baseline_cache", ROOT / "scripts/ci/ppu/cuda_ci_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "hub"
        self.cache.mkdir()
        self.model = "ExampleOrg/real-model"
        self.entry = self.cache / "models--ExampleOrg--real-model"

    def probe(self):
        self.assertTrue(hasattr(baseline, "probe_model_cache"), "read-only cache probe is missing")
        return baseline.probe_model_cache(self.model, str(self.cache))

    def test_missing_model_is_not_present(self):
        row = self.probe()
        self.assertEqual(row["cache_status"], "missing")
        self.assertFalse(row["exists"])

    def test_snapshot_symlink_counts_target_files_without_writes(self):
        target = self.root / "checkpoint"
        target.mkdir()
        (target / "config.json").write_text("{}")
        (target / "model.safetensors").write_bytes(b"weights")
        (self.entry / "snapshots").mkdir(parents=True)
        (self.entry / "snapshots/main").symlink_to(target, target_is_directory=True)
        (self.entry / "refs").mkdir()
        (self.entry / "refs/main").write_text("main")
        before = {str(p): p.stat().st_mtime_ns for p in target.iterdir()}
        row = self.probe()
        self.assertEqual(row["cache_status"], "present")
        self.assertTrue(row["exists"])
        self.assertEqual(row["size_bytes"], 9)
        self.assertEqual(before, {str(p): p.stat().st_mtime_ns for p in target.iterdir()})

    def test_dangling_snapshot_is_incomplete(self):
        (self.entry / "snapshots").mkdir(parents=True)
        (self.entry / "snapshots/main").symlink_to(self.root / "missing", target_is_directory=True)
        (self.entry / "refs").mkdir()
        (self.entry / "refs/main").write_text("main")
        row = self.probe()
        self.assertEqual(row["cache_status"], "incomplete")

    def test_empty_model_directory_is_not_available(self):
        self.entry.mkdir()
        row = self.probe()
        self.assertEqual(row["cache_status"], "incomplete")
        self.assertEqual(row["size_bytes"], 0)

    def test_unreadable_cache_root_is_not_reported_as_missing_models(self):
        self.assertTrue(hasattr(baseline, "probe_model_cache"), "read-only cache probe is missing")
        with self.assertRaises((FileNotFoundError, ValueError, OSError)):
            baseline.probe_model_cache(self.model, str(self.root / "unmounted"))

    def snapshot(self):
        path = self.entry / "snapshots/abcdef"
        path.mkdir(parents=True)
        (self.entry / "refs").mkdir()
        (self.entry / "refs/main").write_text("abcdef")
        return path

    def test_no_ref_and_partial_shards_are_incomplete(self):
        path = self.snapshot()
        (path / "config.json").write_text("{}")
        (path / "model-00001-of-00002.safetensors").write_bytes(b"one")
        self.assertEqual(self.probe()["cache_status"], "incomplete")
        (path / "model-00002-of-00002.safetensors").write_bytes(b"two")
        self.assertEqual(self.probe()["cache_status"], "present")
        (self.entry / "refs/main").unlink()
        self.assertEqual(self.probe()["cache_status"], "incomplete")

    def test_dangling_payload_cycles_and_duplicate_links(self):
        path = self.snapshot()
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"weight")
        (path / "same.safetensors").symlink_to(path / "model.safetensors")
        (path / "cycle").symlink_to(path, target_is_directory=True)
        row = self.probe()
        self.assertEqual(row["cache_status"], "present")
        self.assertEqual(row["size_bytes"], 8)
        (path / "tokenizer.json").symlink_to(self.root / "missing")
        self.assertEqual(self.probe()["cache_status"], "incomplete")

    def test_tokenizer_payload_not_config_alone(self):
        path = self.snapshot()
        (path / "tokenizer_config.json").write_text("{}")
        self.assertEqual(self.probe()["cache_status"], "incomplete")
        (path / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {"hello": 0}}}))
        row = self.probe()
        self.assertEqual(row["cache_status"], "present")
        self.assertIn("static", row["reason"].lower())

    def test_complete_peft_adapter_is_present_without_base_model_config(self):
        path = self.snapshot()
        config = path / "adapter_config.json"
        config.write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": "Org/base"}))
        weights = path / "adapter_model.safetensors"
        weights.write_bytes(b"adapter")
        self.assertEqual(self.probe()["cache_status"], "present")
        self.assertIn("adapter", self.probe()["reason"])
        (path / "tokenizer.model").write_bytes(b"tokenizer")
        weights.write_bytes(b"")
        self.assertEqual(self.probe()["cache_status"], "incomplete")
        weights.unlink()
        self.assertEqual(self.probe()["cache_status"], "incomplete")
        weights.write_bytes(b"adapter")
        for value in ({}, {"peft_type": "LORA"}, {"base_model_name_or_path": "Org/base"}, []):
            with self.subTest(config=value):
                config.write_text(json.dumps(value))
                self.assertEqual(self.probe()["cache_status"], "incomplete")

    def test_complete_adapter_ignores_trainer_state_payloads(self):
        path = self.snapshot()
        (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": "Org/base"}))
        (path / "adapter_model.safetensors").write_bytes(b"adapter")
        for name in ("rng_state.pth", "rng_state_0.pth", "scheduler.pt"):
            with self.subTest(training_state=name):
                (path / name).write_bytes(b"training state")
                self.assertEqual(self.probe()["cache_status"], "present")
        (path / "model.safetensors").write_bytes(b"base")
        self.assertEqual(self.probe()["cache_status"], "incomplete")

    def test_adapter_missing_shards_remain_incomplete(self):
        path = self.snapshot()
        (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": "Org/base"}))
        (path / "adapter_model-00001-of-00002.safetensors").write_bytes(b"one")
        self.assertEqual(self.probe()["cache_status"], "incomplete")
        (path / "adapter_model-00002-of-00002.safetensors").write_bytes(b"two")
        self.assertEqual(self.probe()["cache_status"], "present")

    def test_base_config_cannot_authorize_adapter_payload(self):
        path = self.snapshot()
        (path / "config.json").write_text("{}")
        (path / "adapter_model.safetensors").write_bytes(b"adapter")
        adapter = path / "adapter_config.json"
        for with_base_weights in (False, True):
            if with_base_weights:
                (path / "model.safetensors").write_bytes(b"base")
            for value in (None, {}, {"peft_type": "LORA"}, []):
                with self.subTest(base_weights=with_base_weights, adapter_config=value):
                    if value is None:
                        if adapter.exists():
                            adapter.unlink()
                    else:
                        adapter.write_text(json.dumps(value))
                    self.assertEqual(self.probe()["cache_status"], "incomplete")
            adapter.write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": "Org/base"}))
            self.assertEqual(self.probe()["cache_status"], "present")

    def test_adapter_config_cannot_authorize_base_weights(self):
        path = self.snapshot()
        (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": "Org/base"}))
        (path / "adapter_model.safetensors").write_bytes(b"adapter")
        (path / "model.safetensors").write_bytes(b"base")
        self.assertEqual(self.probe()["cache_status"], "incomplete")
        (path / "config.json").write_text("{}")
        self.assertEqual(self.probe()["cache_status"], "present")
        (path / "adapter_model.safetensors").unlink()
        self.assertEqual(self.probe()["cache_status"], "incomplete")

    def test_missing_indexed_weights_not_hidden_by_tokenizer(self):
        path = self.snapshot()
        (path / "config.json").write_text("{}")
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "missing.safetensors"}}))
        (path / "tokenizer.model").write_bytes(b"tokenizer")
        self.assertEqual(self.probe()["cache_status"], "incomplete")

    def test_remote_probe_is_batched_stdin_only_python36_stdlib(self):
        self.assertTrue(hasattr(baseline, "probe_model_caches"))
        source = inspect.getsource(baseline.probe_model_cache)
        ast.parse(source, feature_version=(3, 6))
        real_run = subprocess.run
        calls = []
        def local_remote(command, **kwargs):
            calls.append((command, kwargs))
            self.assertEqual(command[-2:], ["python3", "-"])
            self.assertIn("timeout", kwargs)
            return real_run([sys.executable, "-"], **kwargs)
        with mock.patch.object(baseline.subprocess, "run", side_effect=local_remote):
            rows = baseline.probe_model_caches([self.model, "Org/second"], str(self.cache), cache_ssh="user@host.example")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["cache_status"] for r in rows}, {"missing"})
        for host in ("-oProxyCommand=bad", "host;bad", "user@host\nwhoami"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                baseline.probe_model_caches([self.model], str(self.cache), cache_ssh=host)
        for model in ("../../outside", "/tmp/model", "Org/model/../../other"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                baseline.probe_model_cache(model, str(self.cache))


if __name__ == "__main__":
    unittest.main()
