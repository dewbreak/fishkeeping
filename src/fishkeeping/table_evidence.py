"""Conservative OCR table evidence selection, without inventing a cell structure.

Coordinates remain OCR coordinates. Whole target rows, preceding headers and
notes form atomic evidence groups; a budget may drop a group but never one cell.
This is a layout heuristic, not a table recognizer or a truth/confidence model.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from fishkeeping.retrieval import query_terms, tokenize
from fishkeeping.vision_models import TextRegion

TABLE_SELECTION_VERSION = "ocr-table-row-context-v1"
_QUERY_FILLER = frozenset(
    "value values table tables row rows column columns figure figures entry entries "
    "amount amounts much many show shown reported listed recorded corresponding".split()
)
_HEADER_LABELS = frozenset(
    "item items metric metrics category categories product products account accounts "
    "tank tanks description parameter parameters year years period periods region regions "
    "segment segments indicator indicators measure measures country countries name names".split()
)
_DERIVED = frozenset(
    "sum total average mean difference compare comparison versus change trend combined "
    "highest lowest maximum minimum increase decrease growth percentage ratio".split()
)
_NOTE = re.compile(
    r"(?:\b(?:note|notes|footnote|footnotes|source|sources|unit|units|usd|eur|gbp|jpy|"
    r"thousand|thousands|million|millions|billion|billions|dollars|percent|unaudited)\b"
    r"|mg\s*/\s*l|[$€£¥%†‡]|注[:：]|单位|千元|万元|百万元|亿元|人民币|不包括|包含)",
    re.IGNORECASE,
)
_NUMERIC = re.compile(r"^[\s\d,.+\-−–—()%$€£¥/:'′″*†‡]+(?:[a-zA-Z]{1,4})?$")
_CONTEXT_LABEL = re.compile(
    r"^(?:notes?\b|footnotes?\b|sources?\b|units?\b|amounts?\s+in\b|单位|附注|注[:：])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TableSelection:
    """Indices always address the caller's original sequence, never a sorted copy."""

    indices: list[int]
    groups: list[list[int]]
    requires_review: bool
    warnings: list[str]
    diagnostics: dict = field(default_factory=dict)
    version: str = TABLE_SELECTION_VERSION


def _text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _numeric(value: str) -> bool:
    normalized = _text(value)
    return any(character.isdigit() for character in normalized) and bool(
        _NUMERIC.fullmatch(normalized)
    )


def _rows(regions: Sequence[TextRegion]) -> list[list[int]]:
    """Cluster approximate text baselines; tall/overlapping boxes are reviewed later."""
    rows: list[list[int]] = []
    for index in sorted(range(len(regions)), key=lambda i: (regions[i].box.y1, regions[i].box.x1)):
        box = regions[index].box
        center = (box.y1 + box.y2) / 2
        best = None
        distance = float("inf")
        for row_number, row in enumerate(rows):
            centers = [(regions[i].box.y1 + regions[i].box.y2) / 2 for i in row]
            heights = [regions[i].box.y2 - regions[i].box.y1 for i in row]
            delta = abs(center - statistics.median(centers))
            if delta <= 0.5 * min(box.y2 - box.y1, statistics.median(heights)):
                if delta < distance:
                    best, distance = row_number, delta
        if best is None:
            rows.append([index])
        else:
            rows[best].append(index)
    rows.sort(key=lambda row: min(regions[i].box.y1 for i in row))
    return [sorted(row, key=lambda i: regions[i].box.x1) for row in rows]


def select_table_evidence(
    regions: Sequence[TextRegion],
    question: str,
    *,
    fits: Callable[[list[int]], bool] | None = None,
) -> TableSelection:
    """Keep row/header/note evidence together, calling ``fits`` for complete unions.

    ``fits`` must measure the caller's final serialized payload, including its
    provenance overhead. Omit it to inspect the unbudgeted candidate groups.
    No model calls, numeric conversions, inferred cell text or source mutations.
    """
    if len(regions) > 256:
        raise ValueError("table selection accepts at most 256 OCR regions")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("table question must not be empty")
    if not regions:
        return TableSelection([], [], True, ["No OCR table evidence is available."])
    if len({region.id for region in regions}) != len(regions):
        raise ValueError("table OCR region IDs must be unique")
    rows = _rows(regions)
    warnings = []
    review = False

    def caution(message: str):
        nonlocal review
        review = True
        if message not in warnings:
            warnings.append(message)

    def reading_order(indices):
        return sorted(set(indices), key=lambda i: (regions[i].box.y1, regions[i].box.x1, i))

    row_terms = [set(tokenize(" ".join(regions[i].text for i in row))) for row in rows]
    # A row is a data candidate only when a textual label precedes numeric cells.
    # Year-only headers and generic label headings are not treated as data rows.
    data_rows = []
    for row_number, row in enumerate(rows):
        label = regions[row[0]].text
        label_terms = set(tokenize(label))
        if (
            len(row) >= 2
            and not _numeric(label)
            and not _CONTEXT_LABEL.search(label.strip())
            and not label_terms <= _HEADER_LABELS
            and any(_numeric(regions[i].text) for i in row[1:])
        ):
            data_rows.append(row_number)
    if not data_rows:
        caution(
            "A labeled numeric table body could not be established; retain full OCR for review."
        )
    elif data_rows[0] == 0:
        caution("The table has no established header above its first data row.")
    for row_number in data_rows:
        cells = rows[row_number]
        # A label unknown to the small generic-heading vocabulary may precede
        # year columns (e.g. State | 2023 | 2024). Never silently treat that
        # possible header as an unrelated data row and drop its column meaning.
        values = [_text(regions[i].text) for i in cells[1:]]
        if len(values) >= 2 and all(
            re.fullmatch(r"[0-9]{4}", value) and 1900 <= int(value) <= 2100 for value in values
        ):
            caution("A labeled all-year row may be a header; full table context requires review.")
    first_data = data_rows[0] if data_rows else len(rows)
    header_rows = list(range(first_data))
    header_terms = set().union(*(row_terms[i] for i in header_rows)) if header_rows else set()
    body_set = set(data_rows)
    note_rows = [
        row_number
        for row_number, row in enumerate(rows)
        if row_number not in body_set and _NOTE.search(" ".join(regions[i].text for i in row))
    ]
    terms = set(query_terms(question)) - _QUERY_FILLER
    all_terms = set().union(*row_terms)
    missing = sorted(terms - all_terms - _DERIVED)
    if missing:
        caution("Some question terms are absent from OCR; requested fields may be missing.")
    if terms & _DERIVED or any(
        word in question for word in ("合计", "平均", "增长", "差额", "相比")
    ):
        caution("Arithmetic or comparison requires review of the complete row and column scope.")
    target_terms = terms - header_terms - _DERIVED
    target_rows = [i for i in data_rows if row_terms[i] & target_terms]
    if not target_rows:
        caution("No specific data row matched the question; full-table fallback requires review.")
    # Repeated labels may refer to sections, currencies or periods. Keep every
    # matching row and its section context instead of selecting the first label.
    labels = Counter(_text(regions[rows[i][0]].text) for i in data_rows)
    if any(labels[_text(regions[rows[i][0]].text)] > 1 for i in target_rows):
        caution("Repeated row labels need section or period disambiguation.")
    heights = [region.box.y2 - region.box.y1 for region in regions]
    typical_height = statistics.median(heights)
    if any(height > 1.8 * typical_height for height in heights):
        caution("Tall OCR boxes may span rows; merged-cell structure is not established.")
    for row in rows:
        for left, right in zip(row, row[1:], strict=False):
            if regions[left].box.x2 > regions[right].box.x1:
                caution("Overlapping OCR columns make cell associations ambiguous.")

    # An uncertain query or layout may require rows that lexical matching never
    # selected (especially cross-language questions and arithmetic). A review
    # flag alone must not legitimize an incomplete subset: retain the entire OCR
    # page as one atomic group, or let the budget reject all of it.
    full_table_fallback = review
    groups = []
    if target_rows and not full_table_fallback:
        for target in target_rows:
            context = set(header_rows) | set(note_rows) | {target}
            # Preserve section headings and wrapped labels immediately preceding
            # a target. They are context, not invented header/cell relationships.
            previous = target - 1
            while previous >= first_data and previous not in body_set:
                context.add(previous)
                previous -= 1
            group = reading_order(i for row_number in context for i in rows[row_number])
            if group not in groups:
                groups.append(group)
    else:
        groups = [reading_order(range(len(regions)))]
        caution("Uncertain query or layout requires the complete OCR table evidence group.")

    selected: list[int] = []
    kept_groups = []
    dropped_groups = 0
    for group in groups:
        trial = reading_order([*selected, *group])
        if fits is not None and not fits(trial):
            dropped_groups += 1
            continue
        selected = trial
        kept_groups.append(group)
    if dropped_groups:
        caution("Budget omitted an entire table evidence group; no partial row was retained.")
    if any(regions[i].confidence < 0.75 for i in selected):
        caution("Selected table OCR includes low-confidence text.")
    if not selected:
        caution("No complete table evidence group fits the requested budget.")
    return TableSelection(
        indices=selected,
        groups=kept_groups,
        requires_review=review,
        warnings=warnings,
        diagnostics={
            "version": TABLE_SELECTION_VERSION,
            "selection_mode": "full_table_fallback" if full_table_fallback else "target_rows",
            "row_bands": len(rows),
            "header_bands": len(header_rows),
            "data_bands": len(data_rows),
            "matched_data_bands": len(target_rows),
            "note_bands": len(note_rows),
            "candidate_groups": len(groups),
            "dropped_groups": dropped_groups,
            "missing_question_terms": missing,
            "structure_verified": False,
        },
    )
