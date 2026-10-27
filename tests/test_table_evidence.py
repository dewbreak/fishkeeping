from __future__ import annotations

import pytest

from fishkeeping.table_evidence import TABLE_SELECTION_VERSION, select_table_evidence
from fishkeeping.vision_models import Box, TextRegion


def cell(identifier, text, x, y, width=70, height=20, confidence=0.99):
    return TextRegion(
        id=identifier,
        text=text,
        box=Box(x1=x, y1=y, x2=x + width, y2=y + height),
        confidence=confidence,
    )


def complex_table():
    regions = [
        cell("unit", "Amounts in USD thousands", 0, 0, 540),
        cell("revenue", "Revenue", 100, 30, 170),
        cell("costs", "Costs", 300, 30, 170),
    ]
    for column, text in enumerate(["Region", "2024", "2025", "2024", "2025"]):
        regions.append(cell(f"h{column}", text, column * 100, 60))
    for row, texts in enumerate(
        [["North", "$100", "(25)", "-3.5", "€9"], ["South", "$80", "(11)", "-1.5", "€7"]]
    ):
        for column, text in enumerate(texts):
            regions.append(cell(f"d{row}-{column}", text, column * 100, 100 + row * 40))
    regions.append(cell("note", "Note: figures exclude shipping.", 0, 200, 540))
    return regions


def selected_texts(regions, result):
    return [regions[i].text for i in result.indices]


def test_multilevel_headers_units_footnote_and_target_row_are_preserved():
    regions = complex_table()
    originals = [r.model_dump() for r in regions]
    result = select_table_evidence(regions, "What is North revenue in 2025?")
    texts = selected_texts(regions, result)
    assert {"Amounts in USD thousands", "Revenue", "Costs", "Region", "North"} <= set(texts)
    assert {"$100", "(25)", "-3.5", "€9", "Note: figures exclude shipping."} <= set(texts)
    assert "South" not in texts
    assert not result.requires_review
    assert result.version == TABLE_SELECTION_VERSION
    assert result.diagnostics["header_bands"] == 3
    assert result.diagnostics["structure_verified"] is False
    assert result.diagnostics["selection_mode"] == "target_rows"
    assert len(result.groups) == 1 and set(result.groups[0]) == set(result.indices)
    assert [r.model_dump() for r in regions] == originals


def test_budget_never_retains_a_partial_target_row_or_drops_units():
    regions = complex_table()
    complete = select_table_evidence(regions, "North revenue 2025")
    result = select_table_evidence(
        regions, "North revenue 2025", fits=lambda indices: len(indices) < len(complete.indices)
    )
    assert result.indices == [] and result.groups == []
    assert result.requires_review and result.diagnostics["dropped_groups"] == 1


def test_budget_omits_complete_second_row_group_without_fragmenting_it():
    regions = complex_table()
    single = select_table_evidence(regions, "North revenue 2025")
    result = select_table_evidence(
        regions, "North and South revenue 2025", fits=lambda ids: len(ids) <= len(single.indices)
    )
    assert selected_texts(regions, result) == selected_texts(regions, single)
    assert result.diagnostics["candidate_groups"] == 2
    assert result.diagnostics["dropped_groups"] == 1 and result.requires_review


def test_repeated_labels_keep_both_rows_and_section_context_for_review():
    regions = complex_table()
    regions.append(cell("section", "Europe", 0, 240, 400))
    regions.extend(
        cell(f"repeat-{i}", text, i * 100, 280)
        for i, text in enumerate(["North", "$60", "(5)", "-0.5", "€3"])
    )
    result = select_table_evidence(regions, "Europe North revenue 2025")
    texts = selected_texts(regions, result)
    assert texts.count("North") == 2 and "Europe" in texts
    assert "$100" in texts and "$60" in texts
    assert result.requires_review and any("Repeated" in warning for warning in result.warnings)


def test_uncovered_field_and_header_only_query_require_review():
    regions = complex_table()
    result = select_table_evidence(regions, "North profit")
    assert result.requires_review
    assert result.diagnostics["missing_question_terms"] == ["profit"]
    assert len(result.indices) == len(regions)
    assert result.diagnostics["selection_mode"] == "full_table_fallback"
    header_only = select_table_evidence(regions, "Revenue 2025")
    assert header_only.requires_review
    assert len(header_only.indices) == len(regions)
    assert header_only.diagnostics["matched_data_bands"] == 0


@pytest.mark.parametrize(
    "question", ["North revenue average", "North revenue difference", "North 合计"]
)
def test_derived_questions_do_not_hide_scope_uncertainty(question):
    regions = complex_table()
    result = select_table_evidence(regions, question)
    assert result.requires_review
    assert any("Arithmetic" in warning for warning in result.warnings)
    assert len(result.indices) == len(regions)
    assert result.diagnostics["selection_mode"] == "full_table_fallback"


@pytest.mark.parametrize(
    "question",
    ["North revenue sum", "North与全部区域的收入合计是多少？", "North profit"],
)
def test_uncertain_scope_is_one_atomic_group_even_when_a_target_row_matches(question):
    regions = complex_table()
    # A second independent table need not lexically match the question at all.
    regions.extend(
        [
            cell("second-header", "Breakdown", 0, 250),
            cell("second-year", "2025", 100, 250),
            cell("second-label", "Adjusted", 0, 290),
            cell("second-value", "42", 100, 290),
        ]
    )
    complete = select_table_evidence(regions, question)
    assert complete.requires_review
    assert complete.diagnostics["matched_data_bands"] > 0
    assert complete.diagnostics["selection_mode"] == "full_table_fallback"
    assert set(complete.indices) == set(range(len(regions)))
    assert complete.groups == [complete.indices]
    attempted = []

    def fits(indices):
        attempted.append(indices)
        return len(indices) < len(regions)

    rejected = select_table_evidence(regions, question, fits=fits)
    assert attempted == [complete.indices]
    assert rejected.indices == rejected.groups == []
    assert rejected.requires_review and rejected.diagnostics["dropped_groups"] == 1


def test_suspected_rowspan_and_overlapping_cells_require_review():
    regions = complex_table()
    regions[8] = regions[8].model_copy(update={"box": Box(x1=0, y1=100, x2=80, y2=160)})
    result = select_table_evidence(regions, "North revenue 2025")
    assert result.requires_review and any("Tall" in warning for warning in result.warnings)
    assert len(result.indices) == len(regions)
    regions = complex_table()
    regions[9] = regions[9].model_copy(update={"box": Box(x1=50, y1=100, x2=170, y2=120)})
    result = select_table_evidence(regions, "North revenue 2025")
    assert result.requires_review and any("Overlapping" in warning for warning in result.warnings)
    assert len(result.indices) == len(regions)


def test_low_confidence_in_header_is_not_hidden_by_confident_value():
    regions = complex_table()
    regions[1] = regions[1].model_copy(update={"confidence": 0.2})
    result = select_table_evidence(regions, "North revenue 2025")
    assert result.requires_review
    assert any("low-confidence" in warning for warning in result.warnings)


def test_split_note_with_numeric_marker_is_not_mistaken_for_a_data_row():
    regions = complex_table()
    regions.extend(
        [
            cell("note-label", "Note", 0, 240),
            cell("note-number", "2", 100, 240),
            cell("note-text", "Excludes discontinued operations", 200, 240, 350),
        ]
    )
    result = select_table_evidence(regions, "North revenue 2025")
    texts = selected_texts(regions, result)
    assert "Excludes discontinued operations" in texts
    assert "2" in texts and "Note" in texts
    assert result.diagnostics["data_bands"] == 2


def test_original_indices_work_with_unsorted_input():
    regions = complex_table()
    expected = selected_texts(regions, select_table_evidence(regions, "North revenue 2025"))
    reversed_regions = list(reversed(regions))
    result = select_table_evidence(reversed_regions, "North revenue 2025")
    assert selected_texts(reversed_regions, result) == expected


@pytest.mark.parametrize("heading", ["State", "Cultivar", "Treatment"])
def test_unknown_labeled_year_header_is_never_discarded_as_an_unrelated_data_row(heading):
    regions = [
        cell("title", "Rice Yield", 0, 0, 300),
        cell("heading", heading, 0, 40),
        cell("previous-year", "2023", 100, 40),
        cell("current-year", "2024", 200, 40),
        cell("unit", "pounds per acre", 100, 70, 180),
        cell("target", "California", 0, 110),
        cell("previous-value", "8800", 100, 110),
        cell("current-value", "8650", 200, 110),
        cell("other", "Other", 0, 150),
        cell("other-value", "7500", 100, 150),
    ]
    result = select_table_evidence(regions, "California yield")
    assert result.requires_review
    assert result.diagnostics["selection_mode"] == "full_table_fallback"
    assert set(result.indices) == set(range(len(regions)))
    assert any("all-year" in warning for warning in result.warnings)
    rejected = select_table_evidence(
        regions, "California yield", fits=lambda ids: len(ids) < len(regions)
    )
    assert rejected.indices == rejected.groups == []


def test_unlabeled_numeric_grid_and_missing_header_require_review():
    regions = [
        cell(str(i), text, i % 2 * 100, i // 2 * 40)
        for i, text in enumerate(["2024", "2025", "12", "15"])
    ]
    result = select_table_evidence(regions, "2025")
    assert len(result.indices) == 4 and result.requires_review
    no_header = complex_table()[8:18]
    result = select_table_evidence(no_header, "North")
    assert result.requires_review and any("no established header" in w for w in result.warnings)


def test_empty_invalid_and_excessive_inputs():
    assert select_table_evidence([], "price").requires_review
    regions = complex_table()
    with pytest.raises(ValueError, match="empty"):
        select_table_evidence(regions, " ")
    with pytest.raises(ValueError, match="unique"):
        select_table_evidence([regions[0], regions[0]], "price")
    with pytest.raises(ValueError, match="256"):
        select_table_evidence([regions[0]] * 257, "price")
