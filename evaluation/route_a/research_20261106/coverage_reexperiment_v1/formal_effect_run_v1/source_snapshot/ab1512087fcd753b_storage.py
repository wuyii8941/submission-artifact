"""Route A event and content-addressed artifact storage."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict


class EventStore:
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "programs").mkdir(exist_ok=True)
        (self.root / "evidence").mkdir(exist_ok=True)
        self.events_path = self.root / "events.jsonl"

    def checkpoint(self, payload: Dict[str, Any], name: str = 'checkpoint.json') -> None:
        path = self.root / name
        temporary = path.with_suffix('.partial')
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def put_text(self, text: str, suffix: str = ".py") -> Dict[str, Any]:
        digest = hashlib.sha256(text.encode()).hexdigest()
        path = self.root / "programs" / (digest + suffix)
        if not path.exists():
            temp = path.with_suffix(path.suffix + ".partial")
            with temp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temp), str(path))
        return {"sha256": digest, "path": str(path)}

    def put_json(self, payload: Any, prefix: str = "evidence") -> Dict[str, Any]:
        text = json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n"
        digest = hashlib.sha256(text.encode()).hexdigest()
        path = self.root / "evidence" / (prefix + "-" + digest + ".json")
        if not path.exists():
            temp = path.with_suffix(".partial")
            with temp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temp), str(path))
        return {"sha256": digest, "path": str(path)}

    def append(self, payload: Dict[str, Any], *, sync: bool = False) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
            handle.flush()
            if sync:
                os.fsync(handle.fileno())
