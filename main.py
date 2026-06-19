"""Run the full lite-path Medallion pipeline as a DAG. Zero-key, DuckDB-only.

    python main.py              # full reload (existing behaviour)
    python main.py --date 2026-06-01  # idempotent backfill for one date (Extension exercise 4)

Stages: extract -> validate(gate) -> transform(dedup -> gold) -> report.
Prints the dedup count (the hook payoff) and the quarantine count.
"""
import argparse
import duckdb

from pipeline import config
from pipeline.dag import DAG
from pipeline.extract import extract_to_bronze
from pipeline.validate import validate, write_quarantine
from pipeline.transform import write_silver, write_gold
from pipeline.load import read_gold


def build_dag(con: duckdb.DuckDBPyConnection, date: str | None = None) -> DAG:
    dag = DAG()

    @dag.task("extract")
    def _extract():
        n = extract_to_bronze(con, date=date)
        return con.execute(f"SELECT * FROM {config.BRONZE}").fetchdf()

    @dag.task("validate", upstream=["extract"])
    def _validate(bronze_df):
        clean, bad = validate(bronze_df)
        n_bad = write_quarantine(bad)
        return {"clean": clean, "n_quarantined": n_bad}

    @dag.task("transform", upstream=["validate"])
    def _transform(v):
        stats = write_silver(con, v["clean"])
        n_gold = write_gold(con)
        return {**stats, "gold_rows": n_gold, "n_quarantined": v["n_quarantined"]}

    @dag.task("report", upstream=["transform"])
    def _report(t):
        return t

    return dag


def main(date: str | None = None) -> dict:
    """Run the pipeline.  date=None means full reload; a YYYY-MM-DD string
    activates idempotent partition backfill (Extension exercise 4 — §14)."""
    config.WAREHOUSE.unlink(missing_ok=True)
    con = duckdb.connect(str(config.WAREHOUSE))
    try:
        results = build_dag(con, date=date).run()
        stats = results["report"]
        print("=== Day 17 pipeline (lite) ===")
        if date:
            print(f"  backfill date       : {date}  (idempotent partition mode)")
        print(f"  bronze rows in      : {stats['rows_in']}")
        print(f"  duplicates dropped  : {stats['dropped_dupes']}  (Silver dedup)")
        print(f"  records quarantined : {stats['n_quarantined']}  (failed the gate)")
        print(f"  silver rows         : {stats['rows_out']}")
        print(f"  gold daily rows     : {stats['gold_rows']}")
        print("\nGold (completed orders by day):")
        print(read_gold(con).to_string(index=False))
        return stats
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 17 Medallion pipeline (lite path)")
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help=(
            "Idempotent backfill: process only orders with created_at = DATE. "
            "Re-running for the same date never duplicates Bronze rows. "
            "Omit for a full reload (default behaviour)."
        ),
    )
    args = parser.parse_args()
    main(date=args.date)
