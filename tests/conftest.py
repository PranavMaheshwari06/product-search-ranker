import shutil
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "esci_sample.parquet"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A scratch PSR_HOME that holds a copy of the committed fixture."""
    dest = tmp_path / "tests" / "fixtures"
    dest.mkdir(parents=True)
    shutil.copy(FIXTURE, dest / FIXTURE.name)
    monkeypatch.setenv("PSR_HOME", str(tmp_path))
    return tmp_path
