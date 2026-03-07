from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DUCKDB_PATH = Path(os.getenv("DUCKDB_PATH", LOG_DIR / "research.duckdb"))
BACKUP_STATE_DIR = Path(os.getenv("BACKUP_STATE_DIR", LOG_DIR / "backups"))
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET = os.getenv("R2_BUCKET", "").strip()
R2_PREFIX = os.getenv("R2_PREFIX", "trade-scout").strip().strip("/")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_backup_files() -> list[Path]:
    files: list[Path] = []
    if LOG_DIR.exists():
        files.extend(sorted(LOG_DIR.glob("*.csv")))
        files.extend(sorted(LOG_DIR.glob("*.json")))
    if DUCKDB_PATH.exists():
        files.append(DUCKDB_PATH)
    return [p for p in files if p.exists() and p.is_file()]


def main() -> None:
    if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET]):
        raise RuntimeError("R2 env vars are not fully configured")

    BACKUP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d/%H%M%S")
    uploaded: list[dict] = []

    for path in iter_backup_files():
        key = f"{R2_PREFIX}/{stamp}/{path.name}"
        client.upload_file(str(path), R2_BUCKET, key)
        uploaded.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_of_file(path),
                "r2_key": key,
            }
        )
        print(f"uploaded {path} -> s3://{R2_BUCKET}/{key}")

    manifest = {
        "created_at_utc": now.isoformat(),
        "bucket": R2_BUCKET,
        "prefix": R2_PREFIX,
        "files": uploaded,
    }
    manifest_path = BACKUP_STATE_DIR / "latest_backup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
