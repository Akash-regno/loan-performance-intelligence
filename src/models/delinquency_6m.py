"""
src/models/delinquency_6m.py
-----------------------------
Import alias so both models live as separate importable modules,
matching the folder structure in the implementation plan.
"""
from src.models.delinquency_3m import Delinquency6mModel as Delinquency6mModel  # noqa: F401

__all__ = ["Delinquency6mModel"]
