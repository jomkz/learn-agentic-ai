"""Tests for plain functions in mcp_server.py."""

from __future__ import annotations

import json

from mcp_server import list_files, read_file, run_query


def test_list_files_valid_path():
    result = list_files("/tmp")
    parsed = json.loads(result)
    assert isinstance(parsed, list)


def test_list_files_invalid_path():
    result = list_files("/nonexistent/path/xyz")
    assert result.startswith("Error")


def test_read_file_valid(tmp_path):
    f = tmp_path / "sample.txt"
    content = "hello world"
    f.write_text(content, encoding="utf-8")
    result = read_file(str(f))
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
