"""Regression guards for previously observed NEO Hunyuan workflow failures.

These tests intentionally protect old fixes when newer errors are patched.
They are cheap static contract tests and run before any GPU is rented.
"""
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "scripts/run_hunyuan_vast.py").read_text()
DOWNLOADER = (ROOT / "scripts/hf_download_with_progress.py").read_text()
WORKFLOW = (ROOT / ".github/workflows/hunyuan-t2v-16gb.yml").read_text()
LEDGER = json.loads((ROOT / "hunyuan-failure-ledger.json").read_text())


class HunyuanRegressionTests(unittest.TestCase):
    def test_slow_host_threshold_is_not_relaxed(self):
        self.assertIn("threshold_MBps=65", RUNNER)
        self.assertIn("below the strict 65 MB/s minimum", RUNNER)
        self.assertIn('default=65.0', DOWNLOADER)

    def test_slow_host_is_blacklisted_before_retry(self):
        self.assertIn("persist_blacklist(selected,measured)", RUNNER)
        self.assertIn("SLOW_HOST_RETRY_REQUIRED", RUNNER)
        self.assertIn("RENTING_NEXT_GPU_HOST", RUNNER)

    def test_inconclusive_probe_is_not_permanently_blacklisted(self):
        self.assertIn("NETWORK_PROBE_UNAVAILABLE_RETRY_REQUIRED", RUNNER)
        self.assertIn("PRE_DOWNLOAD_SPEED_TEST_TIMEOUT", RUNNER)
        self.assertIn("without permanently blacklisting this host", RUNNER)

    def test_retry_uses_shared_deadline(self):
        self.assertIn("NEO_GPU_DEADLINE_MONOTONIC", RUNNER)
        self.assertIn("left=deadline-time.monotonic()", RUNNER)
        self.assertIn("os.execv(sys.executable", RUNNER)

    def test_host_blacklist_is_loaded_before_offer_selection(self):
        self.assertIn('blacklist_path=Path("gpu-host-blacklist.json")', RUNNER)
        self.assertIn("if is_blacklisted(o):", RUNNER)
        self.assertIn("blacklist_path.write_text", RUNNER)

    def test_ssh_endpoint_fallback_remains_enabled(self):
        self.assertIn('source="instance_api" if index==0', RUNNER)
        self.assertIn('else "vastai_ssh_url"', RUNNER)
        self.assertIn('"nvidia-smi"', RUNNER)
        self.assertIn("SSH_ENDPOINTS_NOT_READY_YET", RUNNER)

    def test_workflow_passes_github_context(self):
        self.assertIn("GITHUB_TOKEN: ${{ github.token }}", WORKFLOW)
        self.assertIn("GITHUB_REPOSITORY: ${{ github.repository }}", WORKFLOW)

    def test_real_hf_transfer_speed_is_independently_checked(self):
        self.assertIn("HF_REAL_DOWNLOAD_SPEED_MBPS", DOWNLOADER)
        self.assertIn("SLOW_HF_DOWNLOAD_HOST", DOWNLOADER)
        self.assertIn("args.min_mbps", DOWNLOADER)

    def test_every_ledger_guard_has_a_test(self):
        test_names = {name for name in dir(self) if name.startswith("test_")}
        for item in LEDGER["known_failures"]:
            for guard in item["regression_guards"]:
                self.assertIn(guard, test_names, f"Missing regression guard {guard} for run {item['run']}")

    def test_ledger_preserves_the_current_failure_and_policy(self):
        self.assertEqual(LEDGER["last_reviewed_run"], 85)
        self.assertTrue(any(item["run"] == 85 for item in LEDGER["known_failures"]))
        self.assertTrue(any("65 MB/s" in rule for rule in LEDGER["invariants"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
