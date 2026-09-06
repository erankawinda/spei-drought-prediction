from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def json_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def implementation_hashes() -> dict[str, str]:
    """Hash the package and dependency declarations that produced an artifact."""
    package = Path(__file__).resolve().parent
    repository_root = package.parents[1]
    files = sorted(package.glob("*.py"))
    files.extend(
        path
        for path in (
            repository_root / "pyproject.toml",
            repository_root / "requirements.txt",
            repository_root / "requirements-dev.txt",
        )
        if path.exists()
    )
    return {
        path.relative_to(repository_root).as_posix(): file_sha256(path)
        for path in files
    }
