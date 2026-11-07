"""Best-schedule store: tuned schedules per kernel signature (anvil.tune.measure.Task.signature = kernel IR, schedule
Target, GpuParams, mem_latency), as JSON so they can be committed and replayed without searching.

    db = BestDB()                         # reads the committed anvil/tune/best/*.json, then the user store
    e = db.get(sig)                       # {"kernel", "op", "candidate", "trace", "compile_kw", "cycles", ...}
    db.put(sig, result.to_entry()); db.save()          # writes the user store (.anvil-cache/tune/best.json)

Entries in later files override earlier ones; `put` keeps the faster of an existing and a new entry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

COMMITTED = Path(__file__).with_name("best")


def user_store() -> Path:
    from ..runtime.rtl import CACHE
    return Path(os.environ.get("ANVIL_TUNE_DB", CACHE / "tune" / "best.json"))


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return dict(d.get("entries", {}))


def save_entries(path: Path, entries: dict, note: str = ""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"version": 1, "note": note, "entries": {k: entries[k] for k in sorted(entries)}}
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


class BestDB:
    def __init__(self, paths=None, write: Path | str | None = None, *, committed: bool = True):
        self.write = Path(write) if write is not None else user_store()
        reads = (sorted(COMMITTED.glob("*.json")) if committed else []) + [Path(p) for p in (paths or [])]
        if self.write not in reads:
            reads.append(self.write)
        self.entries = {}
        for p in reads:
            self.entries.update(_load(Path(p)))
        self.dirty = {}

    def get(self, sig):
        return self.entries.get(sig)

    def put(self, sig, entry: dict) -> bool:
        old = self.entries.get(sig)
        if old is not None and old.get("cycles") is not None and (entry.get("cycles") is None
                                                                   or old["cycles"] <= entry["cycles"]):
            return False
        self.entries[sig] = entry
        self.dirty[sig] = entry
        return True

    def save(self):
        if not self.dirty:
            return
        cur = _load(self.write)
        cur.update(self.dirty)
        save_entries(self.write, cur)
        self.dirty = {}
