"""
Generalisable evaluation module for comparing data extracted by LLMs with
data extracted by hand.
"""

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

import sklearn.metrics  # type:ignore[import-untyped]
from loguru import logger
from rapidfuzz import fuzz
from rich.console import Console
from rich.table import Table

from deet.data_models.base import Attribute, AttributeType, GoldStandardAnnotation
from deet.data_models.documents import (
    GoldStandardAnnotatedDocument,
    GoldStandardAnnotatedDocumentList,
)
from deet.data_models.evaluation import AttributeMetric, RunMetricsReport
from deet.evaluators.metrics import (
    METRICS,
    EvaluationMetricSettings,
    MetricFunction,
    check_metric_returns_float,
    get_metrics_for_attribute_type,
)
from deet.evaluators.source_fidelity import (
    classify_match_status,
    is_gold_value_in_text,
)
from deet.exceptions import DuplicateAnnotationError, MissingDocumentError
from deet.processors.eppi_citation_parser import (
    format_parsed_citations,
    parse_eppi_citations_from_details,
)
from deet.utils.text_normalisation import normalize_string_for_match

# Default for ``short_snippet_max_len``: snippets shorter than this (in characters)
# use stricter matching—digit-boundary checks for all-numeric snippets, else
# substring or partial fuzzy—so tiny phrases are not scored like full sentences.
_DEFAULT_SHORT_SNIPPET_MAX_LEN = 4


def _verbatim_fuzzy_match_pct(
    snippet_text: str | None,
    document_context: str | None,
    *,
    short_snippet_max_len: int = _DEFAULT_SHORT_SNIPPET_MAX_LEN,
) -> float:
    """
    Score how well a short verbatim snippet is grounded in document text.

    Compares **snippet_text** (needle, e.g. EPPI or LLM ``additional_text``) against
    **document_context** (haystack, usually the LLM annotated document's ``context``).
    Returns a 0-100 similarity-style score.

    For snippets at least ``short_snippet_max_len`` characters long, uses
    :func:`rapidfuzz.fuzz.partial_ratio` (best local alignment in the long context).
    For shorter **all-numeric** snippets (e.g. counts like ``"32"``), uses a stricter
    number-boundary regex so a small number is not conflated with digits inside a
    larger run (e.g. ``"321"``). For other short snippets, prefers substring match,
    else partial ratio.

    Args:
        snippet_text: Verbatim snippet to locate (e.g. human or model
            ``additional_text``).
        document_context: Full document text to search within.
        short_snippet_max_len: Character length below which the snippet is treated as
            "short" for the stricter heuristics described above.

    Returns:
        Float in ``[0.0, 100.0]``, or ``0.0`` if either input is empty.

    """
    # Preserve case for grounding: PDF context and snippets should match as written.
    normalized_snippet = normalize_string_for_match(
        snippet_text or "", case_insensitive=False
    )
    normalized_context = normalize_string_for_match(
        document_context or "", case_insensitive=False
    )
    if not normalized_snippet or not normalized_context:
        return 0.0
    # Short all-numeric snippet: require a standalone number, not a substring of digits.
    if (
        len(normalized_snippet) < short_snippet_max_len
        and normalized_snippet.isdecimal()
    ):
        if re.search(
            r"(?<![0-9])" + re.escape(normalized_snippet) + r"(?![0-9])",
            normalized_context,
        ):
            return 100.0
        return 0.0
    # Other short snippets: exact substring is full credit; else partial fuzzy match.
    if len(normalized_snippet) < short_snippet_max_len:
        return (
            100.0
            if normalized_snippet in normalized_context
            else float(
                fuzz.partial_ratio(normalized_snippet, normalized_context),
            )
        )
    return float(fuzz.partial_ratio(normalized_snippet, normalized_context))


def _eppi_full_text_details_colon_separated(annotation: object) -> str:
    """
    Join all non-empty ``Text`` values from ``item_attribute_full_text_details``.

    EPPI may attach several fragments; for CSV export we concatenate them into one
    cell using ``": "`` as a readable separator (not an EPPI-native format—avoids
    commas inside the cell and keeps the column single-valued).

    Non-EPPI annotations (no list on the model) yield an empty string.
    """
    details = getattr(annotation, "item_attribute_full_text_details", None) or []
    parts: list[str] = []
    for d in details:
        text = getattr(d, "text", None)
        if text is not None and str(text).strip():
            parts.append(str(text).strip())
    return ": ".join(parts)


def _citation_fields_from_annotation(
    annotation: GoldStandardAnnotation,
) -> tuple[str, str]:
    """
    Parse EPPI citation markup into ``(citation_page, citation_highlight_text)``.

    Non-EPPI annotations (no ``item_attribute_full_text_details``) yield empty
    strings. Multiple detail entries are joined with ``": "``.
    """
    details = getattr(annotation, "item_attribute_full_text_details", None) or []
    return format_parsed_citations(parse_eppi_citations_from_details(details))


@dataclass(slots=True)
class EvaluationRow:
    """Container for per-document values needed by extraction/source scoring."""

    gold_value: Any
    predicted_value: Any | None
    citation_haystack: str
    context: str | None


class GoldStandardLLMEvaluator:
    """
    A class to manage the evaluation of LLM-extracted data against
    "gold-standard" ground truth data.
    """

    def __init__(  # noqa: PLR0913
        self,
        gold_standard_annotated_documents: Sequence[GoldStandardAnnotatedDocument],
        llm_annotated_documents: Sequence[GoldStandardAnnotatedDocument],
        attributes: Sequence[Attribute],
        extraction_run_id: str,
        custom_metrics: list[str] | None = None,
        metric_settings: EvaluationMetricSettings | None = None,
    ) -> None:
        """
        Initialise GoldStandardLLMEvaluator with a list of ground truth and
        LLM-generated data to compare, along with the attributes you want to
        compare.

        Args:
            gold_standard_annotated_documents: Human / gold annotations.
            llm_annotated_documents: LLM annotations to score.
            attributes: Attributes to evaluate.
            extraction_run_id: Run identifier written into metric rows.
            custom_metrics: Optional sklearn metric names to merge in.
            metric_settings: Thresholds for extraction metrics (e.g. edit
                distance). Defaults to :class:`EvaluationMetricSettings`.

        """
        self.gold_standard_annotated_documents = gold_standard_annotated_documents
        self.llm_annotated_documents = GoldStandardAnnotatedDocumentList(
            gold_standard_annotations=llm_annotated_documents
        )
        self.attributes = attributes
        self.extraction_run_id = extraction_run_id
        self.metric_settings = metric_settings or EvaluationMetricSettings()
        self.metrics_config: dict[str, MetricFunction] = METRICS
        self.custom_metrics: dict[str, MetricFunction] = {}
        self.calculated_metrics: list[AttributeMetric] = []
        if custom_metrics is not None:
            self.add_custom_metrics(custom_metrics)

    def add_custom_metrics(self, custom_metrics: list[str]) -> None:
        """Add custom metrics. These must be valid metrics from sklearn.metrics."""
        for custom_metric_name in custom_metrics:
            custom_metric = getattr(sklearn.metrics, custom_metric_name, None)
            if custom_metric is not None:
                if check_metric_returns_float(custom_metric):
                    self.metrics_config[custom_metric_name] = custom_metric
                    self.custom_metrics[custom_metric_name] = custom_metric
                else:
                    logger.warning(
                        f"Tried to add {custom_metric_name} to"
                        " evaluation metrics, but it does not return a float."
                    )
            else:
                logger.warning(
                    f"Tried to add {custom_metric_name} to"
                    " evaluation metrics, but it does not exist"
                )

    def evaluate_llm_annotations(
        self,
    ) -> None:
        """
        Compare a list of human annotations to those generated by llms.
        Return a list of AttributeMetric objects.
        """
        if self.calculated_metrics:
            logger.warning("Already calculated metrics, deleting and overwriting.")
            self.calculated_metrics = []
        for attribute in self.attributes:
            logger.debug(
                f"Calculating metric for attribute: {attribute.attribute_label}"
            )
            rows = self._collect_attribute_rows(attribute)
            y_true = [row.gold_value for row in rows]
            y_pred = [row.predicted_value for row in rows]

            applicable_metrics = get_metrics_for_attribute_type(
                attribute.output_data_type,
                settings=self.metric_settings,
            )
            combined_metrics = {**applicable_metrics, **self.custom_metrics}

            self._append_count_metric(
                attribute=attribute,
                metric_name="n_gold_instances",
                metric_value=float(len(rows)),
            )
            self._append_metrics_for_subset(
                attribute=attribute,
                y_true=y_true,
                y_pred=y_pred,
                metrics=combined_metrics,
                suffix="",
            )

            if attribute.output_data_type not in {
                AttributeType.STRING,
                AttributeType.INTEGER,
                AttributeType.FLOAT,
            }:
                continue

            in_context_flags = [
                is_gold_value_in_text(
                    gold_value=row.gold_value,
                    haystack_text=row.context,
                    attribute_type=attribute.output_data_type,
                    edit_distance_threshold=self.metric_settings.edit_distance_match_threshold,
                    allow_string_near_match=False,
                )
                for row in rows
            ]
            in_citation_flags = [
                is_gold_value_in_text(
                    gold_value=row.gold_value,
                    haystack_text=row.citation_haystack,
                    attribute_type=attribute.output_data_type,
                    edit_distance_threshold=self.metric_settings.edit_distance_match_threshold,
                    allow_string_near_match=True,
                )
                for row in rows
            ]
            good_indices = [i for i, is_good in enumerate(in_context_flags) if is_good]
            bad_indices = [
                i for i, is_good in enumerate(in_context_flags) if not is_good
            ]

            self._append_count_metric(
                attribute=attribute,
                metric_name="n_good_source_instances",
                metric_value=float(len(good_indices)),
            )
            self._append_count_metric(
                attribute=attribute,
                metric_name="n_good_citation_instances",
                metric_value=float(sum(in_citation_flags)),
            )

            self._append_metrics_for_subset(
                attribute=attribute,
                y_true=[y_true[i] for i in good_indices],
                y_pred=[y_pred[i] for i in good_indices],
                metrics=combined_metrics,
                suffix="_given_good_source",
            )
            self._append_metrics_for_subset(
                attribute=attribute,
                y_true=[y_true[i] for i in bad_indices],
                y_pred=[y_pred[i] for i in bad_indices],
                metrics=combined_metrics,
                suffix="_given_bad_source",
            )

    def _append_count_metric(
        self,
        *,
        attribute: Attribute,
        metric_name: str,
        metric_value: float,
    ) -> None:
        """Append a count-like metric row."""
        self.calculated_metrics.append(
            AttributeMetric(
                attribute=attribute,
                metric_name=metric_name,
                value=metric_value,
                extraction_run_id=self.extraction_run_id,
            )
        )

    def _append_metrics_for_subset(
        self,
        *,
        attribute: Attribute,
        y_true: list[Any],
        y_pred: list[Any | None],
        metrics: dict[str, MetricFunction],
        suffix: str,
    ) -> None:
        """
        Run metrics over one subset and append rows.

        Empty stratified subsets leave ``value=None`` so the CSV cell is blank
        rather than a misleading ``0.0`` (e.g. ``edit_distance_match_rate``
        returns ``0.0`` for empty lists).
        """
        empty_subset = bool(suffix) and not y_true
        for metric_name, metric_fn in metrics.items():
            metric_key = f"{metric_name}{suffix}"
            if empty_subset:
                value = None
            else:
                try:
                    value = float(metric_fn(y_true, y_pred))
                except (ValueError, TypeError) as error:
                    logger.warning(
                        f"Metric '{metric_key}' not applicable for "
                        f"attribute '{attribute.attribute_label}' "
                        f"(type={attribute.output_data_type}): {error}"
                    )
                    value = None
            self.calculated_metrics.append(
                AttributeMetric(
                    attribute=attribute,
                    metric_name=metric_key,
                    value=value,
                    extraction_run_id=self.extraction_run_id,
                )
            )

    def _collect_attribute_rows(self, attribute: Attribute) -> list[EvaluationRow]:
        """Collect gold/prediction/source context rows for one attribute."""
        rows: list[EvaluationRow] = []
        for document in self.gold_standard_annotated_documents:
            doc_id = document.document.safe_identity.document_id
            logger.debug(f"Extracting gold/LLM values for doc {doc_id}")
            gs_val = document.get_attribute_annotation(attribute).output_data
            gold_real = next(
                (
                    ann
                    for ann in document.annotations
                    if ann.attribute.attribute_id == attribute.attribute_id
                ),
                None,
            )
            citation_haystack = ""
            if gold_real is not None:
                additional_text = str(gold_real.additional_text or "")
                detail_text = _eppi_full_text_details_colon_separated(gold_real)
                _citation_page, citation_highlight = _citation_fields_from_annotation(
                    gold_real
                )
                citation_haystack = " ".join(
                    part
                    for part in (additional_text, citation_highlight, detail_text)
                    if part.strip()
                )

            context: str | None = None
            llm_val: Any | None = None
            try:
                llm_doc = self.llm_annotated_documents.get_by_id(doc_id)
                context = (
                    str(text)
                    if (text := llm_doc.document.context) not in (None, "")
                    else None
                )
            except MissingDocumentError:
                logger.warning(f"LLM annotated doc not found - ID: {doc_id}")
                rows.append(
                    EvaluationRow(
                        gold_value=gs_val,
                        predicted_value=None,
                        citation_haystack=citation_haystack,
                        context=context,
                    )
                )
                continue

            try:
                llm_val = llm_doc.get_attribute_annotation(attribute).output_data
            except DuplicateAnnotationError:
                logger.warning(
                    f"LLM produced multiple annotations for a single"
                    f" attribute with doc: {doc_id}"
                )
                llm_val = None

            rows.append(
                EvaluationRow(
                    gold_value=gs_val,
                    predicted_value=llm_val,
                    citation_haystack=citation_haystack,
                    context=context,
                )
            )
        return rows

    def display_metrics(self) -> None:
        """Print metrics in a nice table to the command line."""
        console = Console()

        table = Table(title="Evaluation Results")

        metric_names = sorted({str(m.metric_name) for m in self.calculated_metrics})

        # Create table with metrics as columns
        table = Table(title="Evaluation Metrics")
        table.add_column("Attribute")
        for name in metric_names:
            table.add_column(name, justify="right")

        # Group metrics by attribute
        metrics_sorted = sorted(
            self.calculated_metrics, key=lambda m: m.attribute.attribute_label
        )
        for attribute, group in groupby(
            metrics_sorted, key=lambda m: m.attribute.attribute_label
        ):
            row = [attribute]
            group_metrics = {str(m.metric_name): m.value for m in group}
            # fill cells in order of metric_names
            row += [
                f"{group_metrics[name]:.4f}"
                if group_metrics.get(name) is not None
                else "N/A"
                for name in metric_names
            ]
            table.add_row(*row)

        console.print(table)

    def write_metrics_to_csv(self, filepath: Path) -> None:
        """Save metrics to csv in wide format (one row per attribute)."""
        if filepath.suffix != ".csv":
            bad_filetype = "file ending must be .csv"
            raise ValueError(bad_filetype)

        base_columns = [
            "extraction_run_id",
            "attribute_id",
            "attribute_label",
            "attribute_type",
        ]
        preferred_metric_columns = [
            "n_gold_instances",
            "n_good_source_instances",
            "n_good_citation_instances",
            "accuracy",
            "accuracy_given_good_source",
            "accuracy_given_bad_source",
            "precision",
            "recall",
            "f1_score",
            "n_labels",
            "edit_distance_match_rate",
            "edit_distance_match_rate_given_good_source",
            "edit_distance_match_rate_given_bad_source",
            "mean_absolute_error",
            "mean_absolute_error_given_good_source",
            "mean_absolute_error_given_bad_source",
            "mean_absolute_percentage_error",
            "mean_absolute_percentage_error_given_good_source",
            "mean_absolute_percentage_error_given_bad_source",
        ]

        by_attribute: dict[tuple[int, str], dict[str, Any]] = {}
        metric_names: set[str] = set()
        for metric in self.calculated_metrics:
            attr_key = (
                metric.attribute.attribute_id,
                metric.attribute.attribute_label,
            )
            if attr_key not in by_attribute:
                by_attribute[attr_key] = {
                    "extraction_run_id": self.extraction_run_id,
                    "attribute_id": metric.attribute.attribute_id,
                    "attribute_label": metric.attribute.attribute_label,
                    "attribute_type": metric.attribute.output_data_type.value,
                }
            by_attribute[attr_key][metric.metric_name] = metric.value
            metric_names.add(metric.metric_name)

        ordered_metric_columns = [
            name for name in preferred_metric_columns if name in metric_names
        ]
        ordered_metric_columns.extend(
            sorted(name for name in metric_names if name not in ordered_metric_columns)
        )
        fieldnames = base_columns + ordered_metric_columns

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writeheader()
            for _, row_values in sorted(
                by_attribute.items(), key=lambda item: item[1]["attribute_label"]
            ):
                row = {name: row_values.get(name, "") for name in fieldnames}
                for metric_name in ordered_metric_columns:
                    if row[metric_name] is None:
                        row[metric_name] = ""
                writer.writerow(row)

    def write_metrics_to_json(self, filepath: Path) -> None:
        """Save metrics to JSON via :class:`RunMetricsReport`."""
        if filepath.suffix != ".json":
            bad_filetype = "file ending must be .json"
            raise ValueError(bad_filetype)
        report = RunMetricsReport.from_attribute_metrics(
            extraction_run_id=self.extraction_run_id,
            calculated_metrics=self.calculated_metrics,
        )
        report.to_json(filepath)

    def export_llm_comparison(
        self,
        filepath: Path,
    ) -> None:
        """
        Export a comparison CSV for gold vs LLM per document and attribute.

        Columns include identifiers, EPPI-oriented fields, extractions, LLM verbatim,
        fuzzy grounding scores (against the LLM annotated document's ``context``),
        and run id.

        Column semantics:

        - ``attribute_presence``: Whether the gold annotation is present.
        - ``human_additional_text`` / ``item_attribute_full_text_details``: Taken from
          the eppi json file when present; empty when absent.
        - ``citation_page`` / ``citation_highlight_text``: Parsed from raw EPPI
          citation markup (``Page N:`` / ``[¬s]...[¬e]``); multiple fragments joined
          with ``": "``. Empty when markup is absent.
        - ``human_extraction``: Actual ground truth to be extracted.
        - ``human_verbatim_fuzzy_match_pct``: Grounding of ``human_additional_text``
          against the LLM annotated document's ``context``.
        - ``llm_verbatim_text`` / ``llm_verbatim_fuzzy_match_pct``: LLM
          ``additional_text`` and its grounding against the same ``context``.

        Example row (illustrative types): ``attribute_presence`` is the string
        ``"True"`` or ``"False"``; ``human_verbatim_fuzzy_match_pct`` and
        ``llm_verbatim_fuzzy_match_pct`` are decimal strings (e.g. ``"100.00"``,
        ``"87.50"``); ``human_extraction`` / ``llm_extraction`` serialize according to
        the attribute's coerced value (e.g. bool, int, or str) as written by
        :class:`csv.DictWriter`.
        """
        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "document_id",
                    "external_id",
                    "document_name",
                    "attribute_id",
                    "attribute_label",
                    "attribute_presence",
                    "human_additional_text",
                    "item_attribute_full_text_details",
                    "citation_page",
                    "citation_highlight_text",
                    "human_extraction",
                    "llm_extraction",
                    "llm_reasoning",
                    "llm_verbatim_text",
                    "human_verbatim_fuzzy_match_pct",
                    "llm_verbatim_fuzzy_match_pct",
                    "gold_value_in_citation",
                    "gold_value_in_context",
                    "match_status",
                    "extraction_run_id",
                ],
            )
            writer.writeheader()
            for doc in self.gold_standard_annotated_documents:
                try:
                    llm_annotated_doc = self.llm_annotated_documents.get_by_id(
                        doc.document.safe_identity.document_id
                    )
                except MissingDocumentError:
                    llm_annotated_doc = None

                context: str | None = (
                    None
                    if llm_annotated_doc is None
                    else (str(t) if (t := llm_annotated_doc.document.context) else None)
                )

                for attribute in self.attributes:
                    human_ann = doc.get_attribute_annotation(attribute)
                    gold_real = next(
                        (
                            ann
                            for ann in doc.annotations
                            if ann.attribute.attribute_id == attribute.attribute_id
                        ),
                        None,
                    )
                    if gold_real is not None:
                        human_additional_text: str = gold_real.additional_text or ""
                        item_attr_full: str = _eppi_full_text_details_colon_separated(
                            gold_real
                        )
                        citation_page, citation_highlight = (
                            _citation_fields_from_annotation(gold_real)
                        )
                    else:
                        human_additional_text = ""
                        item_attr_full = ""
                        citation_page = ""
                        citation_highlight = ""
                    present = gold_real is not None
                    human_fuzzy = _verbatim_fuzzy_match_pct(
                        human_additional_text, context
                    )

                    llm_extraction: Any = None
                    llm_reasoning: str | None = None
                    llm_verbatim: str | None = None
                    llm_fuzzy = 0.0

                    if llm_annotated_doc is None:
                        llm_reasoning = (
                            "LLM did not produce an output for this document."
                            " Check the logs carefully to find out why"
                        )
                    else:
                        try:
                            llm_annotation = llm_annotated_doc.get_attribute_annotation(
                                attribute
                            )
                            llm_extraction = llm_annotation.output_data
                            llm_reasoning = llm_annotation.reasoning
                            llm_verbatim = llm_annotation.additional_text
                            llm_fuzzy = _verbatim_fuzzy_match_pct(llm_verbatim, context)
                        except DuplicateAnnotationError:
                            llm_reasoning = (
                                "The LLM produced multiple annotations"
                                "for this single attribute"
                            )

                    if attribute.output_data_type in {
                        AttributeType.STRING,
                        AttributeType.INTEGER,
                        AttributeType.FLOAT,
                    }:
                        gold_in_citation = is_gold_value_in_text(
                            gold_value=human_ann.output_data,
                            haystack_text=" ".join(
                                part
                                for part in (
                                    human_additional_text,
                                    citation_highlight,
                                    item_attr_full,
                                )
                                if part.strip()
                            ),
                            attribute_type=attribute.output_data_type,
                            edit_distance_threshold=self.metric_settings.edit_distance_match_threshold,
                            allow_string_near_match=True,
                        )
                        gold_in_context = is_gold_value_in_text(
                            gold_value=human_ann.output_data,
                            haystack_text=context,
                            attribute_type=attribute.output_data_type,
                            edit_distance_threshold=self.metric_settings.edit_distance_match_threshold,
                            allow_string_near_match=False,
                        )
                        match_status = classify_match_status(
                            gold_value=human_ann.output_data,
                            predicted_value=llm_extraction,
                            gold_in_context=gold_in_context,
                            attribute_type=attribute.output_data_type,
                            edit_distance_threshold=self.metric_settings.edit_distance_match_threshold,
                        )
                    else:
                        gold_in_citation = None
                        gold_in_context = None
                        match_status = None

                    writer.writerow(
                        {
                            "document_id": doc.document.safe_identity.document_id,
                            "external_id": doc.document.safe_identity.external_id,
                            "document_name": doc.document.name,
                            "attribute_id": attribute.attribute_id,
                            "attribute_label": attribute.attribute_label,
                            "attribute_presence": str(present),
                            "human_additional_text": human_additional_text,
                            "item_attribute_full_text_details": item_attr_full,
                            "citation_page": citation_page,
                            "citation_highlight_text": citation_highlight,
                            "human_extraction": human_ann.output_data,
                            "llm_extraction": llm_extraction,
                            "llm_reasoning": llm_reasoning,
                            "llm_verbatim_text": llm_verbatim,
                            "human_verbatim_fuzzy_match_pct": f"{human_fuzzy:.2f}",
                            "llm_verbatim_fuzzy_match_pct": f"{llm_fuzzy:.2f}",
                            "gold_value_in_citation": (
                                ""
                                if gold_in_citation is None
                                else str(gold_in_citation)
                            ),
                            "gold_value_in_context": (
                                "" if gold_in_context is None else str(gold_in_context)
                            ),
                            "match_status": match_status or "",
                            "extraction_run_id": self.extraction_run_id,
                        }
                    )

    def export_llm_csv(self, filepath: Path) -> None:
        """Write the LLM output to csv."""
        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "document_id",
                    "external_id",
                    "document_name",
                    "attribute_id",
                    "attribute_label",
                    "llm_extraction",
                    "llm_reasoning",
                    "llm_verbatim_text",
                    "extraction_run_id",
                ],
            )
            writer.writeheader()
            for (
                llm_annotated_doc
            ) in self.llm_annotated_documents.gold_standard_annotations:
                for attribute in self.attributes:
                    try:
                        llm_annotation = llm_annotated_doc.get_attribute_annotation(
                            attribute
                        )
                        llm_extraction = llm_annotation.output_data
                        llm_reasoning = llm_annotation.reasoning
                        llm_verbatim = llm_annotation.additional_text
                    except DuplicateAnnotationError:
                        llm_extraction = None
                        llm_reasoning = (
                            "The LLM produced multiple annotations"
                            "for this single attribute"
                        )
                        llm_verbatim = None

                    document = llm_annotated_doc.document

                    writer.writerow(
                        {
                            "document_id": document.safe_identity.document_id,
                            "external_id": document.safe_identity.external_id,
                            "document_name": document.name,
                            "attribute_id": attribute.attribute_id,
                            "attribute_label": attribute.attribute_label,
                            "llm_extraction": llm_extraction,
                            "llm_reasoning": llm_reasoning,
                            "llm_verbatim_text": llm_verbatim,
                            "extraction_run_id": self.extraction_run_id,
                        }
                    )
