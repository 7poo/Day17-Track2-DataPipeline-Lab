"""E of ELT: load the raw CSV into the Bronze layer, append-only, unchanged.

Design choice: Bronze is the immutable landing zone. We never edit it — if a
downstream layer is wrong, we can always rebuild from Bronze. This is why
'always keep the raw layer' is Rule #1 of Medallion architecture.
"""
import duckdb
from . import config


def extract_to_bronze(con: duckdb.DuckDBPyConnection, date: str | None = None) -> int:
    """Load raw CSV into Bronze.

    date=None (default): full reload — drops and recreates the table.
    date='YYYY-MM-DD':   partition-aware idempotent mode (Extension exercise 4).
        - Creates the table if it does not exist yet.
        - DELETEs existing rows for that date, then re-INSERTs from the CSV.
        - Re-running for the same date never duplicates rows.
        - Other dates are untouched, so backfill windows can overlap safely.
    """
    csv_pos = config.RAW_CSV.as_posix()
    if date is None:
        # Original full-reload behaviour — existing tests rely on this path.
        con.execute(f"DROP TABLE IF EXISTS {config.BRONZE}")
        con.execute(
            f"""
            CREATE TABLE {config.BRONZE} AS
            SELECT * FROM read_csv_auto('{csv_pos}', header=true, all_varchar=true)
            """
        )
    else:
        # Idempotent partition backfill: delete-then-reinsert for the target date.
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.BRONZE} AS
            SELECT * FROM read_csv_auto('{csv_pos}', header=true, all_varchar=true)
            WHERE 1 = 0
            """
        )
        con.execute(f"DELETE FROM {config.BRONZE} WHERE created_at = '{date}'")
        con.execute(
            f"""
            INSERT INTO {config.BRONZE}
            SELECT * FROM read_csv_auto('{csv_pos}', header=true, all_varchar=true)
            WHERE created_at = '{date}'
            """
        )
    (n,) = con.execute(f"SELECT count(*) FROM {config.BRONZE}").fetchone()
    return n
