"""CPU-only contracts for P1 extraction, frozen LPT plans and report CLI."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("cuda_plan", ROOT / "scripts/ci/ppu/cuda_ci_baseline.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)
SHA = "a" * 40
SUITE = "base-a-test-1-gpu-small"


def registration(name, time=10, suite=SUITE, **changes):
    return dict(dict(file="test/registered/" + name + ".py", backend="CUDA", suite=suite,
                     est_time_s=time, disabled="", nightly_flag=False,
                     cadence=baseline.cadence_of(suite), runner_label="1-gpu-h100",
                     n_devices=1, gpu_model="H100", gpu_mem_gb=80), **changes)


class SweepPlanTests(unittest.TestCase):
    def setUp(self):
        self.rows = [registration("test_a", 8), registration("test_b", 7),
                     registration("test_c", 6), registration("test_d", 5)]

    def plan(self, **kwargs):
        self.assertTrue(hasattr(baseline, "build_sweep_plan"), "sweep planner is missing")
        return baseline.build_sweep_plan(self.rows, **dict(dict(suites=SUITE, sha=SHA), **kwargs))

    def test_original_lpt_and_resource_budget(self):
        plan = self.plan(partition_size=2)
        jobs = plan["matrix"]["include"]
        self.assertEqual([j["files"] for j in jobs], [[self.rows[0]["file"], self.rows[3]["file"]], [self.rows[1]["file"], self.rows[2]["file"]]])
        self.assertEqual(plan["max_jobs"], 24)
        self.assertEqual(len(plan["expected"]), 4)
        for j in jobs:
            self.assertEqual((j["board"], j["n_devices"], j["node_selector"]), ("890P", 1, "ZW-M890P"))
            self.assertEqual(j["ppu_supports_fp8"], 1)
            self.assertEqual(j["timeout_per_file"], 600)
            self.assertGreater(j["timeout_minutes"], 20)
            self.assertGreater(j["job_minutes"], j["timeout_minutes"])
            self.assertLessEqual(j["job_minutes"], 360)
            self.assertRegex(j["key"], r"^[a-zA-Z0-9_-]{1,80}$")
        self.assertEqual(len({j["key"] for j in jobs}), 2)

    def test_all_auto_resources_use_890p_and_preserve_cuda_cards(self):
        for memory, cards in ((32, 1), (80, 2), (96, 4), (141, 8), (180, 16), (288, 24)):
            with self.subTest(memory=memory, cards=cards):
                self.rows = [registration("test_board", gpu_mem_gb=memory, n_devices=cards)]
                plan = self.plan()
                job, = plan["matrix"]["include"]
                self.assertEqual((job["board"], job["n_devices"], job["ppu_supports_fp8"]), ("890P", cards, 1))
                self.assertEqual(plan["expected"][0]["ppu_supports_fp8"], 1)
                report, = baseline.build_sweep_suites(self.rows)
                self.assertEqual((report["board"], report["n_devices"], report["ppu_supports_fp8"]), ("890P", cards, 1))
                blocked = self.plan(blocked_files={self.rows[0]["file"]: "missing"})
                self.assertEqual(blocked["blocked"][0]["ppu_supports_fp8"], 1)

    def test_explicit_810e_calibration_and_concurrency_are_not_clamped(self):
        plan = self.plan(board="810E", max_parallel=31)
        self.assertEqual(plan["max_jobs"], 31)
        for item in plan["expected"] + plan["matrix"]["include"]:
            self.assertEqual((item["board"], item["ppu_supports_fp8"]), ("810E", 0))
        self.rows = [registration("test_too_large", gpu_mem_gb=141)]
        with self.assertRaisesRegex(ValueError, "810E"):
            self.plan(board="810E")

    def test_nightly_is_not_a_nightly_flag_and_disabled_excluded(self):
        suite = "nightly-test-1-gpu"
        self.rows = [registration("test_yes", suite=suite), registration("test_no", suite=suite, nightly_flag=True), registration("test_disabled", suite=suite, disabled="broken")]
        self.assertEqual([r["file"] for r in self.plan(suites=suite)["expected"]], [self.rows[0]["file"]])

    def test_board_mapping_preserves_four_cards_and_capacity(self):
        self.rows = [registration("test_b200", 300, n_devices=4, gpu_mem_gb=180, gpu_model="B200")]
        plan = self.plan(max_parallel=16)
        job, = plan["matrix"]["include"]
        self.assertEqual((job["board"], job["n_devices"], job["node_selector"]), ("890P", 4, "ZW-M890P"))
        self.assertEqual(job["timeout_per_file"], 900)
        self.assertEqual(plan["max_jobs"], 4)

    def test_invalid_inputs_fail_visibly(self):
        for args in ({"suites": ""}, {"suites": "unknown"}, {"suites": SUITE + ","}, {"suites": "weekly-test-1-gpu"},
                     {"partition_size": 5}, {"partition_size": 1.5}, {"partition_size": -1},
                     {"max_parallel": 0}, {"max_parallel": "1.5"}, {"max_parallel": 1000000},
                     {"timeout_per_file": -1}, {"timeout_per_file": 3601}, {"timeout_per_file": 1.5},
                     {"board": "H100"}, {"sha": "main"}, {"sha": "A" * 40}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.plan(**args)
        for changes in ({"n_devices": None}, {"gpu_mem_gb": None}, {"n_devices": 25}, {"est_time_s": float("nan")}):
            self.rows = [registration("test_bad", **changes)]
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan()

    def test_missing_model_shards_preserve_lpt_and_do_not_launch(self):
        blocked = {self.rows[0]["file"]: "missing Example/model"}
        with self.assertRaisesRegex(ValueError, "partition.*file_count|singleton"):
            self.plan(partition_size=2, blocked_files=blocked)
        plan = self.plan(partition_size=4, blocked_files=blocked)
        self.assertEqual(len(plan["matrix"]["include"]), 3)
        self.assertEqual(plan["blocked"][0]["auto_status"], "INFRA_MODEL")
        self.assertEqual(len(plan["expected"]), 4)
        self.assertEqual([j["partition_id"] for j in plan["matrix"]["include"]], [1, 2, 3])
        plan = self.plan(blocked_files={r["file"]: "missing" for r in self.rows})
        self.assertEqual(plan["matrix"]["include"], [])
        self.assertEqual(len(plan["blocked"]), 4)

    def test_mixed_duplicates_reject_and_singletons_share(self):
        other = "extra-test-1-gpu-large"
        self.rows += [registration("test_a", 8, suite=other), registration("test_new", 9, suite=other)]
        with self.assertRaisesRegex(ValueError, "partition.*file_count|singleton"):
            self.plan(suites=SUITE + "," + other)
        # Both suites now have four files, allowing a single scalar partition size.
        self.rows += [registration("test_c", 6, suite=other), registration("test_d", 5, suite=other)]
        plan = self.plan(suites=SUITE + "," + other, partition_size=4)
        self.assertEqual(len(plan["matrix"]["include"]), 5)
        self.assertEqual(len(plan["expected"]), 8)

    def test_same_file_with_different_timeout_is_not_deduplicated(self):
        other = "extra-test-1-gpu-large"
        self.rows = [registration("test_a", 10), registration("test_a", 400, suite=other)]
        plan = self.plan(suites=SUITE + "," + other)
        self.assertEqual(len(plan["matrix"]["include"]), 2)
        self.assertEqual([j["timeout_per_file"] for j in plan["matrix"]["include"]], [600, 1200])

    def test_github_job_and_matrix_limits_reject_not_truncate(self):
        self.rows = [registration("test_%03d" % n) for n in range(257)]
        with self.assertRaisesRegex(ValueError, "256"):
            self.plan(partition_size=257)
        with self.assertRaisesRegex(ValueError, "partition"):
            self.plan(timeout_per_file=3600)

    def test_offline_suite_report_retains_unknown_resources(self):
        self.assertTrue(hasattr(baseline, "build_sweep_suites"))
        self.rows.append(registration("test_unknown", suite="extra-test-unknown", n_devices=None, gpu_mem_gb=None))
        rows = baseline.build_sweep_suites(self.rows)
        self.assertEqual(len(rows), 2)
        self.assertTrue(next(r for r in rows if r["suite"] == "extra-test-unknown")["reason"])
        ready = next(r for r in rows if r["suite"] == SUITE)
        self.assertEqual((ready["file_count"], ready["timeout_per_file"]), (4, 600))
        self.assertGreaterEqual(ready["proposed_partitions"], 1)


class ModelExtractionTests(unittest.TestCase):
    def test_dataset_only_references_do_not_block_model_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            file = repo / registration("test_data")["file"]
            file.parent.mkdir(parents=True)
            file.write_text(
                'from huggingface_hub import snapshot_download\n'
                'from sglang.benchmark.utils import download_and_cache_hf_file as fetch\n'
                'MODEL = "Org/model"\nDATA = "Org/data"\nSHARE = "Org/share"\n'
                'snapshot_download(DATA, repo_type="dataset")\n'
                'snapshot_download(repo_id="Org/inline", repo_type="dataset")\n'
                'fetch(SHARE, "samples.json")\n'
                'load_model(MODEL)\n'
            )
            rows = [registration("test_data")]
            inventory = baseline.build_model_inventory(rows, repo_root=repo)
            self.assertEqual(inventory["file_models"][rows[0]["file"]], ["Org/model"])
            self.assertEqual(inventory["file_datasets"][rows[0]["file"]], ["Org/data", "Org/inline", "Org/share"])
            self.assertIn("dataset", inventory["unresolved_files"][rows[0]["file"]])
            cache = repo / "hub"
            snapshot = cache / "models--Org--model/snapshots/main"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            refs = snapshot.parent.parent / "refs"
            refs.mkdir()
            (refs / "main").write_text("main")
            _, blocked = baseline.model_availability(inventory, cache)
            self.assertEqual(blocked, {})
            self.assertEqual(len(baseline.build_sweep_plan(rows, suites=SUITE, sha=SHA, blocked_files=blocked)["matrix"]["include"]), 1)
            file.write_text(file.read_text() + 'load_model(DATA)\n')
            inventory = baseline.build_model_inventory(rows, repo_root=repo)
            self.assertEqual(inventory["file_models"][rows[0]["file"]], ["Org/data", "Org/model"])

    def test_download_helper_respects_positional_repo_type(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            row = registration("test_repo_type")
            file = repo / row["file"]
            file.parent.mkdir(parents=True)
            file.write_text(
                'from sglang.benchmark.utils import download_and_cache_hf_file as fetch\n'
                'fetch("Org/actual-model", "config.json", "model")\n'
                'fetch("Org/data", "samples.json", "dataset")\n'
                'fetch("Org/keyword-model", "config.json", repo_type="model")\n'
                'fetch("Org/dynamic-type", "config.json", selected_type)\n'
            )
            inventory = baseline.build_model_inventory([row], repo_root=repo)
            self.assertEqual(inventory["file_models"][row["file"]], ["Org/actual-model", "Org/dynamic-type", "Org/keyword-model"])
            self.assertEqual(inventory["file_datasets"][row["file"]], ["Org/data"])

    def test_fstring_display_label_is_not_a_model_reference(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            row = registration("test_label")
            file = repo / row["file"]
            file.parent.mkdir(parents=True)
            file.write_text('print(f"{\'triton/cuda\':>12}")\n')
            inventory = baseline.build_model_inventory([row], repo_root=repo)
            self.assertEqual(inventory["file_models"][row["file"]], [])

    def test_fstring_call_arguments_remain_model_references(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            row = registration("test_formatted_call")
            file = repo / row["file"]
            file.parent.mkdir(parents=True)
            file.write_text(
                'MODEL = "Org/plain-model"\n'
                'print(f"{load_model(\'Org/actual-model\')}")\n'
                'print(f"{load_model(model=\'Org/keyword-model\'):>12}")\n'
                'print(f"{\'triton/cuda\':>12}")\n'
            )
            inventory = baseline.build_model_inventory([row], repo_root=repo)
            self.assertEqual(inventory["file_models"][row["file"]], ["Org/actual-model", "Org/keyword-model", "Org/plain-model"])

    def test_static_shared_alias_attribute_comma_lists_and_unknowns(self):
        self.assertTrue(hasattr(baseline, "build_model_inventory"))
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            helper = repo / "python/sglang/test/test_utils.py"
            helper.parent.mkdir(parents=True)
            helper.write_text('SHARED = "Org/shared"\nNIGHTLY = "Org/first,Org/second"\nUNUSED = "Org/unused"\n')
            sources = {
                "test_a": 'from sglang.test.test_utils import SHARED as alias\nx = alias\nx = "Org/inline"\n',
                "test_b": 'from sglang.test import test_utils as tu\nx = tu.NIGHTLY\n',
                "test_c": 'x = "Org/one,Org/two"\n',
                "test_unknown": 'x = get_dynamic_model()\n',
                "test_broken": 'def broken(',
            }
            rows = [registration(name) for name in sources]
            for name, source in sources.items():
                file = repo / registration(name)["file"]
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(source)
            inv = baseline.build_model_inventory(rows, repo_root=str(repo))
            self.assertEqual(inv["file_models"][rows[0]["file"]], ["Org/inline", "Org/shared"])
            self.assertEqual(inv["file_models"][rows[1]["file"]], ["Org/first", "Org/second"])
            self.assertEqual(inv["file_models"][rows[2]["file"]], ["Org/one", "Org/two"])
            self.assertIn(rows[3]["file"], inv["unresolved_files"])
            self.assertIn(rows[4]["file"], inv["parse_failures"])
            overrides = repo / "overrides.json"
            overrides.write_text(json.dumps({"deny": ["Org/inline"], "by_file": {rows[0]["file"]: ["Org/override"]}, "by_suite": {SUITE: ["Org/suite"]}}))
            inv = baseline.build_model_inventory(rows, repo_root=str(repo), overrides_path=str(overrides))
            self.assertEqual(inv["file_models"][rows[0]["file"]], ["Org/override", "Org/shared", "Org/suite"])


class SweepCliTests(unittest.TestCase):
    def test_cli_outputs_inventory_and_scoped_failure_before_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out, evidence, cache = root / "out", root / "evidence", root / "hub"
            evidence.mkdir()
            cache.mkdir()
            rows = [registration("test_requested"), registration("test_unrequested", suite="extra-test-1-gpu-large")]
            def fake_build(directory):
                Path(directory).mkdir(parents=True, exist_ok=True)
                baseline.write_csv(str(Path(directory) / "registrations.csv"), rows)
                baseline.write_csv(str(Path(directory) / "cuda_files.csv"), [{"file": r["file"], "area": "unit", "keyword_hints": "triton", "ppu_suites": "ppu-test"} for r in rows])
                return {"cuda": {"by_cadence": {}}, "ppu": {"enabled_registrations": 0}}
            with mock.patch.object(baseline, "build", side_effect=fake_build), mock.patch.object(baseline, "git_head", return_value=(SHA, "", "")), mock.patch.object(baseline, "REPO", str(root)):
                self.assertEqual(baseline.main(["--out-dir", str(out), "--cache-root", str(cache), "--sweep-plan", "--ref", SHA]), 0)
                for name in ("sweep_plan.json", "model_inventory.json", "model_availability.csv", "model_blocked_files.csv", "sweep_suites.csv"):
                    self.assertTrue((out / name).is_file(), name)
                self.assertEqual(baseline.main(["--out-dir", str(out), "--sweep-results", str(evidence)]), 1,
                                 "an empty artifact tree must not report success")
                plan = json.loads((out / "sweep_plan.json").read_text())
                self.assertEqual(plan["max_jobs"], 24)
                (evidence / "sweep_plan.json").write_text(json.dumps(plan))
                args = ["--out-dir", str(out), "--sweep-results", str(evidence)]
                self.assertEqual(baseline.main(args), 1)
                with (out / "parity_manifest.csv").open() as stream:
                    manifest = list(csv.DictReader(stream))
                self.assertEqual(len(manifest), 2)
                self.assertEqual({r["auto_status"] for r in manifest}, {"NOT_EXECUTED"})
                self.assertEqual(manifest[0]["area"], "unit")
                self.assertTrue((out / "sweep_results.csv").read_text().startswith("file,"))
                worker = evidence / "worker"
                (worker / "junit").mkdir(parents=True)
                f = rows[0]["file"]
                (worker / "sweep.json").write_text(json.dumps(dict(plan["matrix"]["include"][0], sha=SHA, image_digest="", source_root="/source", run_id=plan["run_id"], attempt=plan["attempt"], plan_attempt=plan["attempt"])))
                (worker / "junit/summary.json").write_text(json.dumps({"selected": ["/source/" + f], "files": [{"file": "/source/" + f, "junit": "case.xml", "log": "case.log", "returncode": 0}]}))
                (worker / "junit/case.xml").write_text('<testsuite><testcase name="ok"/></testsuite>')
                (worker / "junit/case.log").write_text("1 passed")
                self.assertEqual(baseline.main(args), 0, "unrequested suites must not fail scoped report")
                # Sharing is scoped to requested tuples, not all inventory registrations.
                rows[1]["file"] = f
                self.assertEqual(baseline.main(args), 0)
                with (out / "parity_manifest.csv").open() as stream:
                    manifest = list(csv.DictReader(stream))
                self.assertEqual([r["auto_status"] for r in manifest], ["PASS", "NOT_EXECUTED"])
                for stage in ("build_model_inventory", "build_parity_manifest"):
                    with mock.patch.object(baseline, stage, side_effect=ValueError("invalid report input")):
                        self.assertEqual(baseline.main(args), 2)
                    for name in ("sweep_results.csv", "parity_manifest.csv"):
                        self.assertNotIn("PASS", (out / name).read_text(), "early and late failures must clear stale PASS")
                    self.assertEqual(baseline.main(args), 0)
                    self.assertFalse((out / "sweep_error.json").exists(), "successful regeneration must clear old errors")
                summary = worker / "junit/summary.json"
                original = json.loads(summary.read_text())
                summary.write_text(json.dumps(dict(original, selected=[])))
                self.assertEqual(baseline.main(args), 1)
                for malformed in ({"files": [{}]}, {"files": None}, []):
                    summary.write_text(json.dumps(malformed))
                    self.assertEqual(baseline.main(args), 2)
                    self.assertNotIn("PASS", (out / "parity_manifest.csv").read_text())
                summary.write_text(json.dumps(original))
                plan_path = evidence / "sweep_plan.json"
                for malformed in ([], {"expected": None}, {"expected": [{}]}):
                    plan_path.write_text(json.dumps(malformed))
                    self.assertEqual(baseline.main(args), 2)
                    self.assertNotIn("PASS", (out / "sweep_results.csv").read_text())
                plan_path.write_text(json.dumps(plan))
                metadata_path = worker / "sweep.json"
                metadata = json.loads(metadata_path.read_text())
                for malformed in ([], {}, dict(metadata, files=None)):
                    metadata_path.write_text(json.dumps(malformed))
                    self.assertEqual(baseline.main(args), 2)
                    self.assertNotIn("PASS", (out / "sweep_results.csv").read_text())
                metadata_path.write_text(json.dumps(metadata))
                (worker / "junit/case.xml").write_text("<broken>")
                self.assertEqual(baseline.main(args), 2)
                for name in ("sweep_results.csv", "parity_manifest.csv"):
                    self.assertNotIn("PASS", (out / name).read_text(), "stale successful output must not survive invalid evidence")
                self.assertEqual(baseline.main(["--out-dir", str(out), "--sweep-plan", "--max-parallel", "0"]), 2)
                self.assertFalse((out / "sweep_plan.json").exists(), "failed replanning must not leave an old dispatch plan")


if __name__ == "__main__":
    unittest.main()
