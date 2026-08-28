import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from p0.synthetic import make_hf, make_lf  # noqa: E402


@pytest.fixture(scope="session")
def hf():
    return make_hf(n_days=4.0, seed=3)


@pytest.fixture(scope="session")
def lf(hf):
    return make_lf(hf)


@pytest.fixture(scope="session")
def store(hf, lf):
    from p0.harness import Store

    return Store(hf, lf)


@pytest.fixture(scope="session")
def folds(store):
    from p0.split import make_folds

    # data 4 ngày từ 2026-01-01: VAL 01-03 và 01-04 (ES = ngày trước), FIT expanding từ origin eligible đầu
    return make_folds(store.first_origin_ts, ["2026-01-03", "2026-01-04"], purge_minutes=60)
