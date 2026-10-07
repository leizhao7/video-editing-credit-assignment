"""Explicit, optional source fixtures for production-source inspection tests."""
import os
from pathlib import Path
import pytest


def source_file(variable, directory=False):
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"Set {variable} to opt into this source/retained-fixture check")
    path = Path(value).expanduser()
    if not (path.is_dir() if directory else path.is_file()):
        pytest.fail(f"Configured {variable} is missing or has the wrong file type")
    return path


def source_root():
    return source_file("OPSD_TEST_SKYRL_ROOT", directory=True)
