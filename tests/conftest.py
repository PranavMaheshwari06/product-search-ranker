import shutil
import zlib
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "esci_sample.parquet"


def fake_encode(texts: list[str]) -> np.ndarray:
    """Hashed bag of words, so shared words mean higher cosine; stands in for e5 offline."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.split(":", 1)[-1].lower().split():
            out[i, zlib.crc32(word.encode()) % 64] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(norms == 0, 1.0, norms)).astype(np.float16)


@pytest.fixture
def fake_dense(monkeypatch):
    """Tests never download the e5 model."""
    from product_search_ranker.retrieval import dense

    monkeypatch.setattr(dense, "encode", fake_encode)
    return dense


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A scratch PSR_HOME that holds a copy of the committed fixture."""
    dest = tmp_path / "tests" / "fixtures"
    dest.mkdir(parents=True)
    shutil.copy(FIXTURE, dest / FIXTURE.name)
    monkeypatch.setenv("PSR_HOME", str(tmp_path))
    return tmp_path
