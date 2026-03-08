# scheduler_tasks.py
from __future__ import annotations

import subprocess
import threading
import time
from datetime import datetime, timezone


def _run(cmd: list[str]) -> None:
    try:
        print(f"[SCHED] running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"[SCHED] command failed: {' '.join(cmd)}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)


def start_scheduler() -> threading.Thread:
    """
    Runs scheduled maintenance inside the same Render service that owns /var/data.
    All times are UTC.
    """
    state = {
        "6h": None,
        "daily": None,
        "duckdb": None,
        "monthly": None,
    }

    def loop() -> None:
        while True:
            now = datetime.now(timezone.utc)

            # every 6 hours
            if now.minute == 0 and now.hour in {0, 6, 12, 18}:
                slot = now.strftime("%Y-%m-%dT%H")
                if state["6h"] != slot:
                    _run(["python", "scripts/backup_to_r2.py", "--mode", "6h"])
                    state["6h"] = slot

            # daily snapshot
            if now.hour == 23 and now.minute == 50:
                slot = now.strftime("%Y-%m-%d")
                if state["daily"] != slot:
                    _run(["python", "scripts/backup_to_r2.py", "--mode", "daily"])
                    state["daily"] = slot

            # nightly DuckDB rebuild
            if now.hour == 23 and now.minute == 40:
                slot = now.strftime("%Y-%m-%d")
                if state["duckdb"] != slot:
                    _run(["python", "scripts/build_research_duckdb.py"])
                    state["duckdb"] = slot

            # monthly archive on the 1st
            if now.day == 1 and now.hour == 0 and now.minute == 10:
                slot = now.strftime("%Y-%m")
                if state["monthly"] != slot:
                    _run(["python", "scripts/backup_to_r2.py", "--mode", "monthly"])
                    state["monthly"] = slot

            time.sleep(30)

    t = threading.Thread(target=loop, daemon=True, name="trade_scout_scheduler")
    t.start()
    return t