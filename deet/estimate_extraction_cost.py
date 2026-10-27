"""Estimate extraction cost from a usage log and prices per million tokens."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from math import fsum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from deet.hierarchical_mvp.extraction_logging import ExtractionUsageLog
from deet.logger import logger

if TYPE_CHECKING:
    from collections.abc import Mapping


class TokenPrices(BaseModel):
    """Rates per million tokens, all in the same caller-selected currency."""

    model_config = ConfigDict(extra="forbid")

    input: Decimal = Field(ge=0, allow_inf_nan=False)
    cached_input: Decimal = Field(ge=0, allow_inf_nan=False)
    output: Decimal = Field(ge=0, allow_inf_nan=False)
    cache_write_input: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)


def estimate_extraction_cost(
    extraction_log: str | Path | Mapping[str, Any] | ExtractionUsageLog,
    token_prices: Mapping[str, Any],
) -> float:
    """
    Return estimated cost; reject unknown usage instead of undercounting.

    Prices are per million tokens. ``cache_write_input`` is the full write rate,
    not a surcharge; when omitted, writes are priced as ordinary input.
    """
    prices = TokenPrices.model_validate(token_prices)
    if isinstance(extraction_log, ExtractionUsageLog):
        log = extraction_log.model_copy(deep=True)
    elif isinstance(extraction_log, (str, Path)):
        log = ExtractionUsageLog.model_validate_json(
            Path(extraction_log).read_text(encoding="utf-8")
        )
    else:
        log = ExtractionUsageLog.model_validate(extraction_log)
    if log.status == "running":
        message = "Cannot estimate an unfinished extraction log."
        raise ValueError(message)
    if log.logging_errors:
        message = "Cannot estimate cost: token logging is incomplete due to errors."
        raise ValueError(message)
    log.summarize()
    totals = log.totals
    if (
        totals.input_tokens is None
        or totals.cached_input_tokens is None
        or totals.output_tokens is None
    ):
        message = "Cannot estimate cost: input, output, or cached usage is unknown."
        raise ValueError(message)
    fresh = totals.input_tokens - totals.cached_input_tokens
    writes = totals.cache_write_input_tokens
    if fresh < 0 or (writes is not None and writes > fresh):
        message = "Cache token counts exceed total input tokens."
        raise ValueError(message)
    input_cost = fresh * prices.input
    if prices.cache_write_input is not None:
        if writes is None:
            message = "Cache-write pricing requires reported cache-write token counts."
            raise ValueError(message)
        input_cost = (fresh - writes) * prices.input + writes * prices.cache_write_input
    else:
        logger.warning(
            "Cache writes are priced as ordinary input; additional cache-write "
            "charges, if applicable, are not included."
        )
    if log.status == "failed":
        logger.warning("Estimating recorded usage for a failed extraction run.")
    return float(
        (
            input_cost
            + totals.cached_input_tokens * prices.cached_input
            + totals.output_tokens * prices.output
        )
        / Decimal(1_000_000)
    )


def summarize_extraction_costs(
    folder_path: str | Path,
    token_prices: Mapping[str, Any],
) -> float:
    """
    Display each log filename and price, then return the overall estimated total.

    Read files starting with ``extraction_log`` directly in the folder, sorted
    by filename. All logs use the same prices per million tokens. Unfinished
    logs are excluded with a warning; other invalid logs raise an error before
    any table or overall total is displayed.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        message = (
            f"Extraction log folder does not exist or is not a directory: {folder}"
        )
        raise NotADirectoryError(message)
    TokenPrices.model_validate(token_prices)
    paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.name.startswith("extraction_log")
    )
    if not paths:
        message = f"No files starting with 'extraction_log' found in {folder}"
        raise FileNotFoundError(message)
    costs: list[tuple[str, float]] = []
    skipped: list[str] = []
    for path in paths:
        try:
            log = ExtractionUsageLog.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if log.status == "running":
                skipped.append(path.name)
                continue
            cost = estimate_extraction_cost(log, token_prices)
        except (OSError, ValueError) as exc:
            message = f"Cannot price extraction log '{path.name}': {exc}"
            raise ValueError(message) from exc
        costs.append((path.name, cost))
    total = fsum(cost for _, cost in costs)
    total_label = (
        "Overall total (excluding unfinished logs)" if skipped else "Overall total"
    )
    width = max([len(total_label), *(len(name) for name, _ in costs)])
    lines = [
        f"Warning: {name} is unfinished; excluded from the total." for name in skipped
    ]
    lines.append(f"{'Name':<{width}}  Price")
    lines.extend(f"{name:<{width}}  {cost:.8f}" for name, cost in costs)
    lines.append(f"{total_label:<{width}}  {total:.8f}")
    print("\n".join(lines))  # noqa: T201
    return total


def main() -> None:
    """Estimate cost using a log path and a JSON price dictionary or file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extraction_log", type=Path)
    prices = parser.add_mutually_exclusive_group(required=True)
    prices.add_argument(
        "--prices", help="JSON dictionary of prices per million tokens."
    )
    prices.add_argument("--prices-file", type=Path, help="JSON price dictionary file.")
    args = parser.parse_args()
    try:
        price_dict = json.loads(
            args.prices_file.read_text(encoding="utf-8")
            if args.prices_file is not None
            else args.prices
        )
        cost = estimate_extraction_cost(args.extraction_log, price_dict)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Cost estimation failed: {exc}\n")
    print(json.dumps({"estimated_cost": cost}))  # noqa: T201


if __name__ == "__main__":
    main()
