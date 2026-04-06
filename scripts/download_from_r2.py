from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DUCKDB_PATH = Path(os.getenv("DUCKDB_PATH", LOG_DIR / "research.duckdb"))
EXECUTION_DB_PATH = Path(os.getenv("EXECUTION_DB_PATH", "./execution.db"))

R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET = os.getenv("R2_BUCKET", "").strip()
R2_PREFIX = os.getenv("R2_PREFIX", "trade-scout").strip().strip("/")


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
    return f"{R2_PREFIX}/latest/"


def list_latest_objects(client) -> list[dict]:
    prefix = latest_prefix()
    paginator = client.get_paginator("list_objects_v2")
    out: list[dict] = []

    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key or key.endswith("/"):
                continue
            out.append(obj)

    return out


def target_path_for_filename(filename: str) -> Path:
    if filename == "research.duckdb":
        return DUCKDB_PATH
    if filename == "execution.db":
        return EXECUTION_DB_PATH
    return LOG_DIR / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download latest Trade Scout research files from R2"
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Filename to download (repeatable), e.g. --only research.duckdb",
    )
    parser.add_argument(
        "--skip-existing-same-size",
        action="store_true",
        help="Skip download if local file exists and size matches remote object",
    )
    return parser.parse_args()


def should_keep_filename(filename: str, only: set[str]) -> bool:
    if not only:
        return True
    return filename in only


def run_download(only: list[str], skip_existing_same_size: bool) -> None:
    client = make_client()
    objects = list_latest_objects(client)

    if not objects:
        print(f"No files found under s3://{R2_BUCKET}/{latest_prefix()}")
        return

    only_set = {name.strip() for name in only if name and name.strip()}

    downloaded: list[tuple[str, Path, int]] = []
    skipped: list[tuple[str, str]] = []

    for obj in objects:
        key = obj["Key"]
        size = int(obj.get("Size", 0))
        filename = Path(key).name

        if not should_keep_filename(filename, only_set):
            skipped.append((filename, "filtered by --only"))
            continue

        target = target_path_for_filename(filename)
        target.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing_same_size and target.exists() and target.is_file():
            local_size = target.stat().st_size
            if local_size == size:
                skipped.append((filename, "same size exists locally"))
                continue

        client.download_file(R2_BUCKET, key, str(target))
        downloaded.append((filename, target, size))
        print(f"downloaded {key} -> {target}")

    if only_set and not downloaded and all(name not in {Path(o['Key']).name for o in objects} for name in only_set):
        print("No matching files found in latest/ for --only values.")

    print("\nDownload summary:")
    print(f"  downloaded: {len(downloaded)}")
    print(f"  skipped:    {len(skipped)}")

    if downloaded:
        print("  files downloaded:")
        for filename, path, size in downloaded:
            print(f"    - {filename} ({size} bytes) -> {path}")

    if skipped:
        print("  files skipped:")
        for filename, reason in skipped:
            print(f"    - {filename}: {reason}")


def main() -> None:
    args = parse_args()
    run_download(args.only, args.skip_existing_same_size)


if __name__ == "__main__":
    main()
