"""Tests for plain functions in mcp_server.py."""

from __future__ import annotations

import json

import pytest

from mobility_ai.phase6.mcp_server import list_files, read_file, run_query


def test_list_files_valid_path(tmp_path):
    result = list_files(".", root=tmp_path)
    parsed = json.loads(result)
    assert isinstance(parsed, list)


def test_list_files_invalid_path():
    result = list_files("/nonexistent/path/xyz")
    assert result.startswith("Error")


def test_read_file_valid(tmp_path):
    f = tmp_path / "sample.txt"
    content = "hello world"
    f.write_text(content, encoding="utf-8")
    result = read_file(str(f), root=tmp_path)
    assert result == content


def test_read_file_missing():
    result = read_file("/no/such/file.txt")
    assert result.startswith("Error")


def test_run_query_select_allowed():
    result = run_query("SELECT * FROM users")
    assert "Query:" in result


def test_run_query_insert_blocked():
    result = run_query("INSERT INTO users VALUES (1)")
    assert result == "Only SELECT queries are allowed"


def test_run_query_update_blocked():
    result = run_query("UPDATE users SET name='x'")
    assert result == "Only SELECT queries are allowed"


def test_run_query_case_insensitive():
    result = run_query("select * from t")
    assert "Query:" in result


def test_run_query_with_whitespace():
    result = run_query("  SELECT id FROM t")
    assert "Query:" in result


@pytest.mark.parametrize("operation", [read_file, list_files])
def test_parent_traversal_blocked(tmp_path, operation):
    root = tmp_path / "allowed"
    root.mkdir()
    assert operation("../outside", root=root).startswith("Error")
    assert operation(str(tmp_path), root=root).startswith("Error")


def test_symlinks_and_symlink_directories_blocked(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("must not leak")
    (root / "link").symlink_to(secret)
    (root / "directory").symlink_to(tmp_path, target_is_directory=True)
    assert read_file("link", root=root).startswith("Error")
    assert read_file("directory/secret.txt", root=root).startswith("Error")
    assert list_files("directory", root=root).startswith("Error")


def test_size_limit_and_environment_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED_ROOT", str(tmp_path))
    (tmp_path / "large.txt").write_text("x" * 3000)
    assert read_file("large.txt") == "x" * 2000


def test_fifo_cannot_block_reader(tmp_path):
    import os

    os.mkfifo(tmp_path / "fifo")
    assert read_file("fifo", root=tmp_path).startswith("Error")
