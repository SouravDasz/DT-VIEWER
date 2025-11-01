"""app.ml package

Provide small helper exports so importing from `app.ml` works as expected.
This file intentionally exposes `greet` from `predict.py` for quick testing.
"""
from .predict import greet

__all__ = ["greet"]
