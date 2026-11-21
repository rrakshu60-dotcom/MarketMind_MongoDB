"""Load the trained models that came with the MarketMind internship project.

Pickle files can run code when they are loaded, so instead of a plain
``joblib.load`` this uses a restricted unpickler: it only lets the file
rebuild objects from the libraries a model legitimately needs (scikit-learn,
CatBoost, NumPy, SciPy, joblib). Anything else, such as ``os.system``, is
refused before it runs.
"""
from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path

from joblib import numpy_pickle

import config

ALLOWED_MODULE_PREFIXES = ("sklearn.", "catboost.", "numpy.", "scipy.", "joblib.")
ALLOWED_EXACT = {
    ("builtins", "slice"), ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "complex"),
    ("builtins", "bytearray"), ("collections", "OrderedDict"), ("copyreg", "_reconstructor"),
    ("builtins", "object"),
}

MODEL_FILES = {
    "forecast": "catboost_daily_model.pkl",
    "kmeans": "kmeans_clustering_model.pkl",
    "churn": "churn_model.pkl",
}


class UnsafeModelError(RuntimeError):
    pass


class ModelNotAvailableError(RuntimeError):
    pass


class _RestrictedUnpickler(numpy_pickle.NumpyUnpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED_EXACT or module == "numpy" or module.startswith(ALLOWED_MODULE_PREFIXES):
            return super().find_class(module, name)
        raise UnsafeModelError(f"Refusing to load {module}.{name} from a model file")


def safe_load(path: Path):
    path = Path(path)
    if not path.exists():
        raise ModelNotAvailableError(f"Model file not found: {path}")
    with open(path, "rb") as fh:
        kwargs = {"mmap_mode": None}
        # joblib >= 1.5 added a required ensure_native_byte_order argument.
        if "ensure_native_byte_order" in inspect.signature(numpy_pickle.NumpyUnpickler.__init__).parameters:
            kwargs["ensure_native_byte_order"] = "auto"
        unpickler = _RestrictedUnpickler(str(path), fh, **kwargs)
        return unpickler.load()


@lru_cache(maxsize=None)
def load_model(name: str):
    """Load one of the bundled models by short name: forecast, kmeans or churn."""
    if name not in MODEL_FILES:
        raise KeyError(f"Unknown model '{name}'. Choose from {sorted(MODEL_FILES)}")
    return safe_load(config.MODELS_DIR / MODEL_FILES[name])
