import os
import sys

# Make tests/reference_wip.py importable from every test module
sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: trains a model on a large dataset (deselect with -m 'not slow')",
    )
