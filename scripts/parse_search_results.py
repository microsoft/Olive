# -----------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -----------------------------------------------------------------------------
"""Parse Olive run logs and collect the run-history search results into a CSV.

Olive prints a run-history summary table near the end of each run log. The table is
rendered with ``tabulate`` in the "grid" format and looks like::

    +------------+-------------------+-------------+----------------+----------------+-----------+
    | model_id   | parent_model_id   | from_pass   | search_point   |   duration_sec | metrics   |
    +============+===================+=============+================+================+===========+
    | 543057fc   |                   |             |                |                | { ... }   |
    +------------+-------------------+-------------+----------------+----------------+-----------+
    | c2c80f63   | 543057fc          | smp         | { ... }        |       0.994844 |           |
    +------------+-------------------+-------------+----------------+----------------+-----------+

The ``search_point`` and ``metrics`` columns each hold a pretty-printed JSON object that
spans several physical lines. Every physical line in the log is prefixed with a timestamp.

This script strips the timestamps, reconstructs the table cells (including the multi-line
JSON), flattens the ``search_point`` parameters and ``metrics`` values into individual
columns, and writes everything to a CSV.

Usage::

    python scripts/parse_search_results.py <log-file-or-folder> [-o results.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Matches the leading "2026-07-24 06:15:44.4962421 " timestamp prefix on every log line.
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ ")

# A grid border line, e.g. "+------+======+------+".
_BORDER_RE = re.compile(r"^\+[-=+]+\+\s*$")

# The header row that identifies the run-history table.
_HEADER_COLUMNS = ["model_id", "parent_model_id", "from_pass", "search_point", "duration_sec", "metrics"]

# Fixed (non-flattened) columns that always come first in the output.
_FIXED_COLUMNS = ["model_id", "parent_model_id", "from_pass", "duration_sec"]

_BLOB_ACCOUNT_URL = "https://oliveaasstorage.blob.core.windows.net"
_BLOB_CONTAINER = "output"


def _strip_timestamp(line: str) -> str:
    """Remove the leading log timestamp from a line, if present."""
    return _TIMESTAMP_RE.sub("", line, count=1)


def _split_row(line: str) -> list[str]:
    """Split a grid content line ("| a | b | c |") into its stripped cell fragments."""
    # Drop the leading and trailing pipe, then split on the remaining pipes.
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dotted-path keys. Empty dicts contribute nothing."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(value, child_prefix))
    else:
        flat[prefix] = obj
    return flat


def _parse_json_cell(text: str) -> dict | None:
    """Parse a reconstructed JSON cell, returning None when the cell is empty/invalid."""
    text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _find_header_index(lines: list[str]) -> int | None:
    """Return the index of the run-history table header line, or None if not found."""
    for idx, line in enumerate(lines):
        stripped = _strip_timestamp(line).strip()
        if stripped.startswith("|") and all(col in stripped for col in _HEADER_COLUMNS):
            return idx
    return None


def _collect_table_rows(lines: list[str], header_idx: int) -> tuple[list[str], list[list[str]]]:
    """Parse the grid table starting at the header line.

    Returns the header column names and a list of rows, where each row is a list of
    reconstructed cell strings (multi-line cells joined with newlines).
    """
    headers = _split_row(_strip_timestamp(lines[header_idx]))

    rows: list[list[str]] = []
    current: list[list[str]] | None = None  # per-column accumulated fragments

    # Start right after the header; the very next border ("+===+") opens the data section.
    for line in lines[header_idx + 1 :]:
        content = _strip_timestamp(line).rstrip("\n")
        stripped = content.strip()

        if _BORDER_RE.match(stripped):
            # A border closes the current logical row (if any) and starts a new one.
            if current is not None:
                rows.append(["\n".join(frag for frag in col if frag).strip() for col in current])
            current = [[] for _ in headers]
            continue

        if stripped.startswith("|"):
            cells = _split_row(content)
            if current is None:
                current = [[] for _ in headers]
            for col_idx, cell in enumerate(cells):
                if col_idx < len(current):
                    current[col_idx].append(cell)
            continue

        # First non-table line marks the end of the table.
        break

    return headers, rows


def _row_to_record(headers: list[str], cells: list[str]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Convert a raw table row into a flattened record dict.

    Returns the record plus the ordered search-parameter and metric column names for the row.
    """
    raw = dict(zip(headers, cells))
    record: dict[str, Any] = {
        "model_id": raw.get("model_id", "").strip(),
        "parent_model_id": raw.get("parent_model_id", "").strip(),
        "from_pass": raw.get("from_pass", "").strip(),
        "duration_sec": raw.get("duration_sec", "").strip(),
    }

    search_point = _parse_json_cell(raw.get("search_point", ""))
    if search_point:
        record.update(_flatten(search_point))

    metrics = _parse_json_cell(raw.get("metrics", ""))
    if metrics:
        record.update(_flatten(metrics))

    return record, sorted(_flatten(search_point or {})), sorted(_flatten(metrics or {}))


def parse_log(path: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Parse a single log file.

    Returns the list of records plus the ordered search-parameter and metric column names
    discovered in this file.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    header_idx = _find_header_index(lines)
    if header_idx is None:
        return [], [], []

    headers, rows = _collect_table_rows(lines, header_idx)

    records: list[dict[str, Any]] = []
    search_cols: list[str] = []
    metric_cols: list[str] = []
    for cells in rows:
        if not any(cell.strip() for cell in cells):
            continue
        record, row_search_cols, row_metric_cols = _row_to_record(headers, cells)
        if not record.get("model_id"):
            continue
        # Skip rows without metrics (e.g. intermediate models that were not evaluated).
        if not row_metric_cols:
            continue
        records.append(record)
        for col in row_search_cols:
            if col not in search_cols:
                search_cols.append(col)
        for col in row_metric_cols:
            if col not in metric_cols:
                metric_cols.append(col)

    return records, search_cols, metric_cols


def write_csv(
    records: list[dict[str, Any]],
    search_cols: list[str],
    metric_cols: list[str],
    output_path: Path,
) -> None:
    """Write the collected records to a CSV file."""
    columns: list[str] = []
    columns.extend(_FIXED_COLUMNS)
    if any("model_size_bytes" in record for record in records):
        columns.extend(["model_size_bytes", "model_size_gb"])
    columns.extend(search_cols)
    columns.extend(metric_cols)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def process_log(log_file: Path, output_path: Path) -> bool:
    """Parse a single log file and write its own CSV. Returns True if any rows were written."""
    records, search_cols, metric_cols = parse_log(log_file)
    if not records:
        logger.warning("No run-history table found in %s", log_file)
        return False

    write_csv(records, search_cols, metric_cols, output_path)
    logger.info("Wrote %d rows to %s", len(records), output_path)
    return True


def _fetch_model_sizes_from_blob(subscription_id: str, job_id: str, model_ids: set[str]) -> dict[str, int]:
    """Return model sizes in bytes from Azure Blob storage for the given job/model IDs.

    Data is expected under: output/<job_id>/cache/default_workflow/runs/<model_id>/...
    """
    # Subscription ID is accepted to align with invocation context for Azure-authenticated runs.
    # Data-plane access uses DefaultAzureCredential and the fixed storage account/container.
    if not subscription_id:
        raise ValueError("subscription_id is required when resolving model sizes from blob storage")

    # Imported lazily so the script can run without the optional azure dependencies
    # when model sizes are not requested.
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError as ex:
        raise ImportError(
            "Resolving model sizes requires the 'azure-identity' and 'azure-storage-blob' packages. "
            "Install them with: pip install azure-identity azure-storage-blob"
        ) from ex

    credential = DefaultAzureCredential()
    blob_service = BlobServiceClient(account_url=_BLOB_ACCOUNT_URL, credential=credential)
    container_client = blob_service.get_container_client(_BLOB_CONTAINER)

    sizes: dict[str, int] = dict.fromkeys(model_ids, 0)

    for model_id in model_ids:
        model_prefix = f"{job_id}/cache/default_workflow/runs/{model_id}/"
        total_size = 0
        for blob in container_client.list_blobs(name_starts_with=model_prefix):
            total_size += int(blob.size or 0)
        sizes[model_id] = total_size

    return sizes


def _enrich_records_with_model_sizes(records: list[dict[str, Any]], subscription_id: str, job_id: str) -> None:
    """Populate model size columns in-place for each record."""
    model_ids = {record.get("model_id", "").strip() for record in records if record.get("model_id")}
    if not model_ids:
        return

    try:
        sizes = _fetch_model_sizes_from_blob(subscription_id, job_id, model_ids)
    except Exception as ex:
        logger.warning("Failed to fetch model sizes from blob storage: %s", ex)
        return

    for record in records:
        model_id = (record.get("model_id") or "").strip()
        size_bytes = sizes.get(model_id)
        if size_bytes is None:
            continue
        record["model_size_bytes"] = size_bytes
        record["model_size_gb"] = round(size_bytes / (1024 * 1024 * 1024), 3)


def process_log_with_blob_sizes(
    log_file: Path,
    output_path: Path,
    subscription_id: str,
    job_id: str,
) -> bool:
    """Parse a log, enrich model sizes from blob, and write CSV."""
    records, search_cols, metric_cols = parse_log(log_file)
    if not records:
        logger.warning("No run-history table found in %s", log_file)
        return False

    _enrich_records_with_model_sizes(records, subscription_id, job_id)
    write_csv(records, search_cols, metric_cols, output_path)
    logger.info("Wrote %d rows to %s", len(records), output_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Path to an Olive log file or a folder containing *.log files.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path when the input is a single log file, or an output directory when the input is a folder. "
            "Each log file always produces its own CSV. Defaults to a '<log-name>.csv' next to each log file."
        ),
    )
    parser.add_argument(
        "--subscription-id",
        type=str,
        default=None,
        help="Azure subscription ID used for job context when resolving model sizes from blob storage.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="OAAS job ID, e.g. op_20260723204831_b48422.",
    )
    args = parser.parse_args()

    if bool(args.subscription_id) != bool(args.job_id):
        parser.error("--subscription-id and --job-id must be provided together.")

    if not args.input.exists():
        parser.error(f"Input path does not exist: {args.input}")

    if args.input.is_dir():
        log_files = sorted(args.input.glob("*.log"))
        if not log_files:
            logger.warning("No *.log files found in %s", args.input)
            return

        output_dir = args.output or args.input
        output_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for log_file in log_files:
            output_csv = output_dir / f"{log_file.stem}.csv"
            if args.subscription_id and args.job_id:
                success = process_log_with_blob_sizes(log_file, output_csv, args.subscription_id, args.job_id)
            else:
                success = process_log(log_file, output_csv)
            if success:
                written += 1
        if not written:
            logger.warning("No search results found in any log file. Nothing written.")
    else:
        if args.output is None:
            output_path = args.input.with_suffix(".csv")
        elif args.output.is_dir() or not args.output.suffix:
            # Output is a directory: write a CSV with the same base name as the input inside it.
            args.output.mkdir(parents=True, exist_ok=True)
            output_path = args.output / f"{args.input.stem}.csv"
        else:
            output_path = args.output
        if args.subscription_id and args.job_id:
            success = process_log_with_blob_sizes(args.input, output_path, args.subscription_id, args.job_id)
        else:
            success = process_log(args.input, output_path)
        if not success:
            logger.warning("No search results found. Nothing written.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
