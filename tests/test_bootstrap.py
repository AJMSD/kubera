from __future__ import annotations

import json

from kubera.bootstrap import bootstrap


def test_bootstrap_creates_workspace_and_snapshot(
    isolated_repo,
    capsys,
) -> None:
    run_context = bootstrap()

    assert run_context.run_directory.exists()
    assert run_context.config_snapshot_path.exists()
    assert run_context.log_file_path.exists()

    payload = json.loads(run_context.config_snapshot_path.read_text(encoding="utf-8"))
    assert payload["ticker"]["symbol"] == "INFY"
    assert payload["providers"]["news_api_key"] is None

    output = capsys.readouterr().out
    assert "Kubera bootstrap ready" in output
    assert run_context.run_id in output
