"""Pytest options for standalone PPU tests.

CI does not go through this file: run_suite.py executes each registered file
as ``python3 <file> -f``, so the Answer test reads its config from the
``SGLANG_PPU_ANSWER_TEST_CONFIG`` environment variable there. The option below
exists for local runs that prefer pytest, e.g.::

    pytest -q -s test/registered/ppu/test_ppu_qwen35_answer.py \\
        --answer-test-config <path to a ppu-answer-test-config/v1 JSON>
"""

import os

ANSWER_TEST_CONFIG_ENV = "SGLANG_PPU_ANSWER_TEST_CONFIG"


def pytest_addoption(parser):
    group = parser.getgroup("ppu-answer")
    group.addoption(
        "--answer-test-config",
        action="store",
        default=None,
        metavar="PATH",
        help="Path to a ppu-answer-test-config/v1 JSON file.",
    )


def pytest_configure(config):
    config_path = config.getoption("--answer-test-config")
    if config_path:
        os.environ[ANSWER_TEST_CONFIG_ENV] = config_path
