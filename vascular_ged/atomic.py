from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _replace(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def write_csv(path: str | Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path = Path(path)

    def write(handle) -> None:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    _replace(path, write)


def write_json(path: str | Path, value: object) -> None:
    _replace(Path(path), lambda handle: json.dump(value, handle, indent=2, sort_keys=True))
