"""Regression checks for Isaac Lab IO API compatibility in Parkour playback."""

from __future__ import annotations

import ast
from pathlib import Path


def test_parkour_play_does_not_import_removed_load_pickle() -> None:
    play_script = (
        Path(__file__).resolve().parents[1]
        / "source/instinctlab/instinctlab/tasks/parkour/scripts/play.py"
    )
    tree = ast.parse(play_script.read_text())

    io_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "isaaclab.utils.io"
        for alias in node.names
    }
    direct_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "load_pickle" not in io_imports
    assert "load_pickle" not in direct_calls
    assert "load_yaml" in io_imports
    assert "load_yaml" in direct_calls


def test_parkour_play_supports_a_free_viewport_camera() -> None:
    play_script = (
        Path(__file__).resolve().parents[1]
        / "source/instinctlab/instinctlab/tasks/parkour/scripts/play.py"
    )
    source = play_script.read_text()
    tree = ast.parse(source)
    argument_flags = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }

    assert "--free_camera" in argument_flags
    assert 'env_cfg.viewer.origin_type = "world"' in source
