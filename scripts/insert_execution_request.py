from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trade_scout.execution.models import ExecutionRequest, ExecutionStatus
from trade_scout.execution.queue import (
    ActiveExecutionRequestExistsError,
    get_connection,
    init_db,
    insert_request,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert a paper execution request")
    parser.add_argument("--alert-id", required=True, help="Alert id to queue for execution")
    parser.add_argument(
        "--db-path",
        default="./execution.db",
        help="Path to the execution SQLite database",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request_id = str(uuid4())
    request = ExecutionRequest(
        request_id=request_id,
        alert_id=args.alert_id,
        requested_ts_utc=utc_now_iso(),
        status=ExecutionStatus.PENDING,
    )

    init_db(args.db_path)

    conn = get_connection(args.db_path)
    try:
        try:
            insert_request(conn, request)
        except ActiveExecutionRequestExistsError:
            print(
                "Failed to insert execution request: "
                f"active request already exists for alert_id={args.alert_id}"
            )
            raise SystemExit(1)
    finally:
        conn.close()

    print(f"Inserted execution request {request_id} for alert {args.alert_id}")


if __name__ == "__main__":
    main()
