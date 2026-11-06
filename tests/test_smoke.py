"""Smoke test: the installed package and its modules import."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    ["sinkhorn_lab", "sinkhorn_lab.solvers", "sinkhorn_lab.metrics"],
)
def test_module_imports(module):
    assert importlib.import_module(module) is not None
