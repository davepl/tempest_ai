#!/usr/bin/env python3
"""Thread-safe in-memory chat store for dashboard comments."""

from __future__ import annotations

import json
from pathlib import Path
import re
import threading
import time
from collections import deque
from typing import Any


class ChatStore:
    """Small bounded chat log with monotonically increasing message IDs."""

    def __init__(self, max_messages: int = 240, persist_path: str | None = None):
        self._lock = threading.Lock()
        self._max_messages = max(10, int(max_messages))
        self._rows: deque[dict[str, Any]] = deque(maxlen=self._max_messages)
        self._next_id = 1
        self._persist_path = Path(persist_path) if persist_path else self._default_persist_path()
        self._load_from_disk()

    @staticmethod
    def _default_persist_path() -> Path:
        scripts_dir = Path(__file__).resolve().parent
        logs_dir = scripts_dir.parent / "logs"
        return logs_dir / "dashboard_chat.jsonl"

    @staticmethod
    def _sanitize_text(text: str) -> str:
        # Enforce one-line text and strip control/bidi spoof characters.
        cleaned = str(text)
        cleaned = cleaned.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        # Remove C0/C1 controls and DEL.
        cleaned = "".join(ch for ch in cleaned if ch >= " " and ch not in {"\x7f"})
        # Remove common bidi overrides/isolates to prevent visual spoofing.
        cleaned = re.sub(r"[\u202a-\u202e\u2066-\u2069]", "", cleaned)
        # Collapse whitespace runs.
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _sanitize_ip(ip: str) -> str:
        raw = str(ip or "").strip()
        if not raw:
            return "unknown"
        # Keep a conservative printable subset for display labels.
        safe = "".join(ch for ch in raw if ch.isalnum() or ch in ".:[]-%")
        safe = safe.strip("%")
        return (safe[:64] if safe else "unknown")

    def _load_from_disk(self) -> None:
        path = self._persist_path
        try:
            if not path.exists():
                return
            max_seen_id = 0
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                msg = self._sanitize_text(row.get("text", ""))
                if not msg:
                    continue
                if len(msg) > 240:
                    msg = msg[:240]
                ip = self._sanitize_ip(row.get("ip", "unknown"))
                try:
                    ts = float(row.get("ts", 0.0) or 0.0)
                except Exception:
                    ts = 0.0
                if ts <= 0.0:
                    ts = float(time.time())
                try:
                    rid = int(row.get("id", 0) or 0)
                except Exception:
                    rid = 0
                if rid > max_seen_id:
                    max_seen_id = rid
                self._rows.append({"id": rid, "ts": ts, "ip": ip, "text": msg})
            self._next_id = max(1, max_seen_id + 1)
        except Exception:
            # Persistence is best-effort; runtime chat remains available.
            return

    def _append_to_disk(self, row: dict[str, Any]) -> None:
        path = self._persist_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
        except Exception:
            # Do not fail chat posting on storage errors.
            return

    def add_message(self, ip: str, text: str) -> dict[str, Any]:
        msg = self._sanitize_text(text)
        if not msg:
            raise ValueError("message_empty")
        if len(msg) > 240:
            raise ValueError("message_too_long")
        with self._lock:
            row = {
                "id": int(self._next_id),
                "ts": float(time.time()),
                "ip": self._sanitize_ip(ip),
                "text": msg,
            }
            self._next_id += 1
            self._rows.append(row)
            self._append_to_disk(row)
            return dict(row)

    def snapshot(self, limit: int = 120) -> list[dict[str, Any]]:
        take = max(1, min(int(limit), self._max_messages))
        with self._lock:
            if not self._rows:
                return []
            return [dict(row) for row in list(self._rows)[-take:]]
