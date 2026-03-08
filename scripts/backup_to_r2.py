from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import boto3
from dotenv import load_dotenv

load_dotenv()

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


def require_r2_env() -> None:
    if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET]):
        raise RuntimeError("R2 env vars are not fully configured")


def make_client():
    require_r2_env()
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def latest_prefix() -> str:
    return f"{R2_PREFIX}/latest"


def six_hour_prefix(now: datetime) -> str:
    stamp = now.strftime("%Y-%m-%d_%H%M")
    return f"{R2_PREFIX}/snapshots/6h/{stamp}"


def daily_prefix(now: datetime) -> str:
    stamp = now.strftime("%Y-%m-%d")
    return f"{R2_PREFIX}/snapshots/daily/{stamp}"


def monthly_prefix(now: datetime) -> str:
    stamp = now.strftime("%Y-%m")
    return f"{R2_PREFIX}/archive/{stamp}"


def upload_snapshot(client, prefix: str, files: Iterable[Path]) -> list[dict]:
    uploaded: list[dict] = []

    for path in files:
        key = f"{prefix}/{path.name}"
        client.upload_file(str(path), R2_BUCKET, key)

        meta = {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_of_file(path),
            "r2_key": key,
        }
        uploaded.append(meta)
        print(f"uploaded {path} -> s3://{R2_BUCKET}/{key}")

    return uploaded


def list_common_prefixes(client, prefix: str) -> list[str]:
    """
    Returns direct child prefixes under prefix.
    Example:
      prefix='trade-scout/snapshots/6h/'
      returns ['trade-scout/snapshots/6h/2026-03-07_0000/', ...]
    """
    paginator = client.get_paginator("list_objects_v2")
    prefixes: list[str] = []

    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix, Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            p = item.get("Prefix")
            if p:
                prefixes.append(p)

    return sorted(prefixes)


def list_all_keys_under_prefix(client, prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []

    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                keys.append(key)

    return keys


def delete_keys(client, keys: list[str]) -> int:
    if not keys:
        return 0

    deleted = 0
    chunk_size = 1000
    for i in range(0, len(keys), chunk_size):
        batch = keys[i:i + chunk_size]
        client.delete_objects(
            Bucket=R2_BUCKET,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        deleted += len(batch)

    return deleted


def prune_snapshot_prefixes(client, base_prefix: str, keep_last: int) -> list[str]:
    """
    Keeps only the newest N snapshot folders directly under base_prefix.
    Example base_prefix:
      trade-scout/snapshots/6h/
      trade-scout/snapshots/daily/
    """
    child_prefixes = list_common_prefixes(client, base_prefix)
    if len(child_prefixes) <= keep_last:
        return []

    to_delete_prefixes = child_prefixes[:-keep_last]
    all_keys: list[str] = []
    for p in to_delete_prefixes:
        all_keys.extend(list_all_keys_under_prefix(client, p))

    deleted_count = delete_keys(client, all_keys)
    print(f"pruned {deleted_count} objects from {len(to_delete_prefixes)} old snapshot folders under {base_prefix}")
    return to_delete_prefixes


def write_manifest(now: datetime, mode: str, uploaded: list[dict], pruned_prefixes: list[str]) -> Path:
    BACKUP_STATE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": now.isoformat(),
        "mode": mode,
        "bucket": R2_BUCKET,
        "prefix": R2_PREFIX,
        "uploaded_files": uploaded,
        "pruned_prefixes": pruned_prefixes,
    }

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    path = BACKUP_STATE_DIR / f"backup_manifest_{mode}_{stamp}.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest {path}")
    return path


def run_backup(mode: str) -> None:
    """
    mode:
      - 6h      -> upload latest + 6h snapshot, prune 6h to keep last 12
      - daily   -> upload latest + daily snapshot, prune daily to keep last 30
      - monthly -> upload latest + monthly archive, no pruning
    """
    now = datetime.now(timezone.utc)
    files = iter_backup_files()
    if not files:
        print("No files found to back up.")
        return

    client = make_client()

    uploaded: list[dict] = []
    pruned_prefixes: list[str] = []

    # always refresh latest/
    uploaded.extend(upload_snapshot(client, latest_prefix(), files))

    if mode == "6h":
        snap_prefix = six_hour_prefix(now)
        uploaded.extend(upload_snapshot(client, snap_prefix, files))
        pruned_prefixes = prune_snapshot_prefixes(
            client,
            base_prefix=f"{R2_PREFIX}/snapshots/6h/",
            keep_last=12,  # 3 days of 6-hour snapshots
        )

    elif mode == "daily":
        snap_prefix = daily_prefix(now)
        uploaded.extend(upload_snapshot(client, snap_prefix, files))
        pruned_prefixes = prune_snapshot_prefixes(
            client,
            base_prefix=f"{R2_PREFIX}/snapshots/daily/",
            keep_last=30,
        )

    elif mode == "monthly":
        snap_prefix = monthly_prefix(now)
        uploaded.extend(upload_snapshot(client, snap_prefix, files))
        # no pruning for archive

    else:
        raise ValueError(f"Unknown mode: {mode}")

    write_manifest(now, mode, uploaded, pruned_prefixes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Back up Trade Scout files to R2")
    parser.add_argument(
        "--mode",
        choices=["6h", "daily", "monthly"],
        required=True,
        help="Backup tier to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_backup(args.mode)


if __name__ == "__main__":
    main()