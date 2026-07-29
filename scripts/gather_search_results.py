# -----------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -----------------------------------------------------------------------------
"""Scan Olive evaluation results in Azure Blob storage and collect them into a CSV.

For each Olive run, the engine caches one JSON file per evaluation under::

    <operation-id>/cache/default_workflow/evaluations/<eval-cache-key>.json

Each file looks like::

    {
        "model_id": "ffae6c42",
        "parent_model_id": "543057fc",
        "search_point": {"index": 3, "smp": {"bits": 4}},
        "signal": {
            "wmt14-en-fr-bleu": {"value": 33.76, "priority": -1, "higher_is_better": true},
            "wmt14-en-fr-ter":  {"value": 54.54, "priority": -1, "higher_is_better": true}
        }
    }

The ``parent_model_id`` and ``search_point`` fields are only present for runs produced by a
recent enough version of Olive; this script tolerates their absence.

This script scans every ``*.json`` file under the evaluations prefix for a given operation id,
flattens the ``search_point`` parameters and ``signal`` metric values into individual columns,
and writes everything to a CSV. Model sizes are resolved from the run cache.

Usage::

    python scripts/gather_search_results.py <operation-id> \
        --subscription-id <sub-id> [-o results.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

# Fixed (non-flattened) columns that always come first in the output.
_FIXED_COLUMNS = ["model_id", "parent_model_id"]

_BLOB_ACCOUNT_URL = "https://oliveaasstorage.blob.core.windows.net"
_BLOB_CONTAINER = "output"


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


def _flatten_signal(signal: dict | None) -> dict[str, Any]:
    """Flatten a MetricResult ``signal`` mapping into ``metric_name -> value`` pairs."""
    flat: dict[str, Any] = {}
    if not isinstance(signal, dict):
        return flat
    for metric_name, sub_result in signal.items():
        # Each sub-result is a SubMetricResult dict ({"value", "priority", "higher_is_better"}).
        if isinstance(sub_result, dict) and "value" in sub_result:
            flat[metric_name] = sub_result["value"]
        else:
            flat[metric_name] = sub_result
    return flat


def _evaluation_to_record(evaluation: dict) -> tuple[dict[str, Any], list[str], list[str]]:
    """Convert one evaluation JSON object into a flattened record dict.

    Returns the record plus the ordered search-parameter and metric column names.
    """
    record: dict[str, Any] = {
        "model_id": (evaluation.get("model_id") or "").strip(),
        "parent_model_id": (evaluation.get("parent_model_id") or "").strip() if evaluation.get("parent_model_id") else "",
    }

    search_point = evaluation.get("search_point")
    search_flat = _flatten(search_point) if isinstance(search_point, dict) else {}
    record.update(search_flat)

    metrics_flat = _flatten_signal(evaluation.get("signal"))
    record.update(metrics_flat)

    return record, sorted(search_flat), sorted(metrics_flat)


def _iter_evaluation_blobs(container_client, operation_id: str):
    """Yield (blob_name, parsed_json) for every evaluation JSON under the operation id."""
    prefix = f"{operation_id}/cache/default_workflow/evaluations/"
    for blob in container_client.list_blobs(name_starts_with=prefix):
        if not blob.name.endswith(".json"):
            continue
        try:
            data = container_client.download_blob(blob.name).readall()
            yield blob.name, json.loads(data)
        except Exception as ex:
            logger.warning("Failed to read evaluation blob %s: %s", blob.name, ex)


def _fetch_model_sizes_from_blob(container_client, operation_id: str, model_ids: set[str]) -> dict[str, int]:
    """Return model sizes in bytes from Azure Blob storage for the given model IDs.

    Data is expected under: <operation-id>/cache/default_workflow/runs/<model_id>/...
    """
    sizes: dict[str, int] = dict.fromkeys(model_ids, 0)
    for model_id in model_ids:
        model_prefix = f"{operation_id}/cache/default_workflow/runs/{model_id}/"
        total_size = 0
        for blob in container_client.list_blobs(name_starts_with=model_prefix):
            total_size += int(blob.size or 0)
        sizes[model_id] = total_size
    return sizes


def _enrich_records_with_model_sizes(container_client, operation_id: str, records: list[dict[str, Any]]) -> None:
    """Populate model size columns in-place for each record."""
    model_ids = {record.get("model_id", "").strip() for record in records if record.get("model_id")}
    if not model_ids:
        return

    try:
        sizes = _fetch_model_sizes_from_blob(container_client, operation_id, model_ids)
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


def scan_evaluations(
    operation_id: str,
    subscription_id: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Scan all evaluation JSON files for an operation id.

    Returns the list of records plus the ordered search-parameter and metric column names.
    """
    if not subscription_id:
        raise ValueError("subscription_id is required when resolving evaluation results from blob storage")

    credential = DefaultAzureCredential()
    blob_service = BlobServiceClient(account_url=_BLOB_ACCOUNT_URL, credential=credential)
    container_client = blob_service.get_container_client(_BLOB_CONTAINER)

    records: list[dict[str, Any]] = []
    search_cols: list[str] = []
    metric_cols: list[str] = []
    for blob_name, evaluation in _iter_evaluation_blobs(container_client, operation_id):
        if not isinstance(evaluation, dict):
            logger.warning("Skipping non-object evaluation blob %s", blob_name)
            continue
        record, row_search_cols, row_metric_cols = _evaluation_to_record(evaluation)
        if not record.get("model_id"):
            continue
        # Skip entries without metrics (e.g. models that were not evaluated).
        if not row_metric_cols:
            continue
        records.append(record)
        for col in row_search_cols:
            if col not in search_cols:
                search_cols.append(col)
        for col in row_metric_cols:
            if col not in metric_cols:
                metric_cols.append(col)

    _enrich_records_with_model_sizes(container_client, operation_id, records)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("operation_id", type=str, help="OAAS operation/job ID, e.g. op_20260723204831_b48422.")
    parser.add_argument(
        "--subscription-id",
        type=str,
        required=True,
        help="Azure subscription ID used for job context when resolving results from blob storage.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path, or an output directory (a '<operation-id>.csv' file is written inside it). "
            "Defaults to '<operation-id>.csv' in the current directory."
        ),
    )
    args = parser.parse_args()

    if args.output is None:
        output_path = Path(f"{args.operation_id}.csv")
    elif args.output.is_dir() or not args.output.suffix:
        args.output.mkdir(parents=True, exist_ok=True)
        output_path = args.output / f"{args.operation_id}.csv"
    else:
        output_path = args.output

    records, search_cols, metric_cols = scan_evaluations(
        args.operation_id,
        args.subscription_id,
    )
    if not records:
        logger.warning("No evaluation results found for operation %s. Nothing written.", args.operation_id)
        return

    write_csv(records, search_cols, metric_cols, output_path)
    logger.info("Wrote %d rows to %s", len(records), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
