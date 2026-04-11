"""Refresh `config/exchange_closures/india.json` from pinned NSE PDF URLs (optional / dev)."""

from __future__ import annotations

import argparse
import io
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests

from kubera.config import load_settings

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - exercised when dev extra not installed
    PdfReader = None  # type: ignore[misc, assignment]


def extract_iso_dates_from_text(text: str) -> list[date]:
    """Best-effort extraction of calendar dates from NSE holiday PDF text."""

    found: set[date] = set()
    for pattern in (
        r"\b(\d{4})-(\d{2})-(\d{2})\b",
        r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b",
    ):
        for m in re.finditer(pattern, text):
            try:
                if len(m.groups()) == 3 and len(m.group(1)) == 4:
                    y, mo, d = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                else:
                    d, mo, y = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                candidate = date(y, mo, d)
            except ValueError:
                continue
            if 2000 <= candidate.year <= 2100:
                found.add(candidate)
    return sorted(found)


def pdf_bytes_to_text(data: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError(
            "pypdf is not installed. Install dev dependencies: pip install -e \".[dev]\""
        )
    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def validate_closures_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("closures file must be a JSON object")
    holidays = payload.get("holidays")
    if not isinstance(holidays, list):
        raise ValueError("closures file must contain a 'holidays' list")
    for raw in holidays:
        date.fromisoformat(str(raw))


def run_check_only(repo_root: Path) -> int:
    settings = load_settings(repo_root)
    path = settings.market.exchange_closures_path
    if not path.exists():
        print(f"Exchange closures file not found (ok for empty optional): {path}")
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_closures_payload(payload)
    print(f"OK: validated {path}")
    return 0


def run_sync_holidays(
    *,
    repo_root: Path | None = None,
    dry_run: bool = False,
    check_only: bool = False,
) -> int:
    settings = load_settings(repo_root)
    root = settings.paths.repo_root
    if check_only:
        return run_check_only(root)

    sync_config_path = root / "config" / "holiday_sync.json"
    if not sync_config_path.exists():
        print(f"Missing {sync_config_path}; nothing to sync.")
        return 1

    sync_config = json.loads(sync_config_path.read_text(encoding="utf-8"))
    sources = sync_config.get("sources")
    if not isinstance(sources, list) or not sources:
        print("config/holiday_sync.json must contain a non-empty 'sources' list.")
        return 1

    merged: set[date] = set()
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        url = entry.get("pdf_url")
        out_rel = entry.get("output")
        if not url or not out_rel:
            print("Skipping source missing pdf_url or output.")
            continue
        print(f"Fetching {url} ...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        text = pdf_bytes_to_text(response.content)
        extracted = extract_iso_dates_from_text(text)
        if len(extracted) < 5:
            print(
                f"Parsed only {len(extracted)} dates from PDF; layout may have changed. "
                "Update extract_iso_dates_from_text or refresh JSON manually."
            )
            return 1
        merged.update(extracted)
        print(f"Extracted {len(extracted)} date tokens from one PDF ({entry.get('id', '?')}).")

    if not merged:
        print("No dates extracted.")
        return 1

    out_path = root / str(sources[0].get("output", "config/exchange_closures/india.json"))
    payload = {
        "schema_version": 1,
        "source": "sync_holidays.py (PDF extraction; verify against NSE/BSE circulars)",
        "as_of": datetime.now(timezone.utc).date().isoformat(),
        "notes": "Regenerated from config/holiday_sync.json. Review before committing.",
        "holidays": [d.isoformat() for d in sorted(merged)],
    }
    if dry_run:
        print(f"Dry run: would write {len(merged)} holidays to {out_path}")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(merged)} holidays).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync exchange closure dates from official PDFs.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate config/exchange_closures/india.json only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and parse but do not write the output JSON.",
    )
    args = parser.parse_args(argv)
    return run_sync_holidays(dry_run=args.dry_run, check_only=args.check_only)


if __name__ == "__main__":
    raise SystemExit(main())
