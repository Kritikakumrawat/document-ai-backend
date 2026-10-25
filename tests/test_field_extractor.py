"""Field extraction from known synthetic OCR text (no image or OCR engine involved)."""

from dataclasses import replace
from datetime import date
from typing import List, Tuple

import pytest

from app.services.field_extractor import DEFAULT_PROFILE, FieldExtractor
from app.services.ocr_service import OcrLine

TODAY = date(2026, 10, 22)


@pytest.fixture
def extractor() -> FieldExtractor:
    return FieldExtractor(today=TODAY)


def values(result) -> Tuple:  # noqa: ANN001
    return tuple(
        getattr(result, name).value if getattr(result, name) else None
        for name in ("full_name", "document_number", "date_of_birth", "expiry_date")
    )


def test_extracts_all_fields_from_labelled_card(extractor: FieldExtractor) -> None:
    text = """
        SYNTHETIC DRIVER LICENSE
        NAME: ALEX SAMPLE
        ID NO: SYN-4821-7730
        DOB: 14/03/1991
        ISS: 15/03/2021  EXP: 14/03/2031
        SYNTHETIC TEST DATA - NOT A REAL ID
    """
    result = extractor.extract_from_text(text)
    assert values(result) == ("ALEX SAMPLE", "SYN-4821-7730", "1991-03-14", "2031-03-14")
    assert result.date_of_birth.method == "label"
    assert result.warnings == []


def test_labels_above_values_are_supported(extractor: FieldExtractor) -> None:
    text = (
        "SURNAME\nSAMPLE\nGIVEN NAMES\nALEX JAMIE\n"
        "DATE OF BIRTH\n1991-03-14\nDATE OF EXPIRY\n2031-03-14\nDOCUMENT NO\nX1234567"
    )
    result = extractor.extract_from_text(text)
    assert values(result) == ("ALEX JAMIE SAMPLE", "X1234567", "1991-03-14", "2031-03-14")
    assert result.full_name.method == "label_given_name_and_surname"
    assert result.expiry_date.method == "label_next_line"


def test_birth_and_expiry_are_told_apart_by_labels_not_order(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("EXPIRES 14 MAR 2031\nBORN 02 AUG 1988")
    assert result.date_of_birth.value == "1988-08-02"
    assert result.expiry_date.value == "2031-03-14"


def test_issue_date_is_never_used_as_birth_or_expiry(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("NAME: ALEX SAMPLE\nDATE OF ISSUE: 15/03/2021")
    assert result.date_of_birth is None
    assert result.expiry_date is None


def test_missing_fields_are_none(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("SYNTHETIC DRIVER LICENSE\nCLASS: C\nRESTRICTIONS: NONE")
    assert values(result) == (None, None, None, None)


def test_bare_numbers_are_not_taken_as_document_numbers(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("4821 7730\n123456789\nPHONE: 5550100")
    assert result.document_number is None


def test_ambiguous_unlabelled_dates_are_not_guessed(extractor: FieldExtractor) -> None:
    # Three recent past dates: none is clearly a birth date, none is in the future.
    result = extractor.extract_from_text("01/02/2020\n03/04/2021\n05/06/2022")
    assert result.date_of_birth is None
    assert result.expiry_date is None


def test_two_unlabelled_future_dates_are_not_guessed(extractor: FieldExtractor) -> None:
    assert extractor.extract_from_text("10/10/2030\n11/11/2031").expiry_date is None


def test_unlabelled_dates_are_used_only_when_unambiguous(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("SOME CARD\n19/04/1990\n15/03/2021\n01/01/2030")
    assert result.date_of_birth.value == "1990-04-19"
    assert result.date_of_birth.method == "unlabeled_oldest_date"
    assert result.expiry_date.value == "2030-01-01"
    assert result.expiry_date.method == "unlabeled_only_future_date"


def test_future_birth_date_is_rejected(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("DOB: 14/03/2035")
    assert result.date_of_birth is None
    assert any("not plausible" in warning for warning in result.warnings)


def test_expiry_before_birth_is_discarded(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("DOB 14/03/1991\nEXP 14/03/1985")
    assert result.date_of_birth.value == "1991-03-14"
    assert result.expiry_date is None
    assert any("not after the date of birth" in warning for warning in result.warnings)


def test_validity_range_uses_the_later_date(extractor: FieldExtractor) -> None:
    assert extractor.extract_from_text("VALIDITY: 01/01/2021 - 01/01/2031").expiry_date.value == "2031-01-01"


def test_ambiguous_day_month_is_flagged(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("DOB: 04/05/1990")
    assert result.date_of_birth.value == "1990-05-04"  # DMY by default
    assert any("day/month" in warning for warning in result.warnings)
    us_result = FieldExtractor(date_order="MDY", today=TODAY).extract_from_text("DOB: 04/05/1990")
    assert us_result.date_of_birth.value == "1990-04-05"


def test_common_ocr_mistakes_are_tolerated(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("NAME; ALEX SAMPLE\nD0B 14/O3/1991\nEXP. 14-03-2031\nID NO: 48O1-77I0")
    assert values(result) == ("ALEX SAMPLE", "4801-7710", "1991-03-14", "2031-03-14")


@pytest.mark.parametrize(
    "line,expected",
    [
        ("LICENSE NO: D1234567", "D1234567"),
        ("4d DLN D1234567", "D1234567"),
        ("EMPLOYEE ID: EMP-004521", "EMP-004521"),
        ("ID#: 99812345", "99812345"),
        ("Document No. X1234567", "X1234567"),
        ("ID NO: SYN 4821 7730", "SYN 4821 7730"),
        ("ID NO: SYN-4821-7730 CLASS: C", "SYN-4821-7730"),
    ],
)
def test_document_number_label_variants(extractor: FieldExtractor, line: str, expected: str) -> None:
    assert extractor.extract_from_text(line).document_number.value == expected


def test_date_after_id_label_is_not_a_document_number(extractor: FieldExtractor) -> None:
    assert extractor.extract_from_text("ID NO: 14/03/2031").document_number is None


def test_unlabelled_document_number_needs_a_distinctive_pattern(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("SYNTHETIC CARD\nSYN-4821-7730")
    assert result.document_number.value == "SYN-4821-7730"
    assert result.document_number.method == "unlabeled_pattern:hyphenated_alphanumeric"


def test_competing_unlabelled_document_numbers_are_not_guessed(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("AB-123456\nCD-987654")
    assert result.document_number is None
    assert any("document numbers" in warning for warning in result.warnings)


@pytest.mark.parametrize("line", ["NAME: DRIVER LICENSE", "NAME: 12345", "NAME: ##"])
def test_invalid_names_are_rejected(extractor: FieldExtractor, line: str) -> None:
    assert extractor.extract_from_text(line).full_name is None


def test_name_stops_before_trailing_field_numbers(extractor: FieldExtractor) -> None:
    assert extractor.extract_from_text("NAME: ALEX SAMPLE 4d").full_name.value == "ALEX SAMPLE"


def test_only_surname_is_returned_with_a_warning(extractor: FieldExtractor) -> None:
    result = extractor.extract_from_text("SURNAME: SAMPLE")
    assert result.full_name.value == "SAMPLE"
    assert result.full_name.method == "label_surname_only"
    assert any("incomplete" in warning for warning in result.warnings)


def _lines(rows: List[Tuple[str, int]]) -> List[OcrLine]:
    """(text, height) rows stacked vertically, like OCR output."""
    lines, top = [], 0
    for text, height in rows:
        lines.append(OcrLine(text=text, confidence=0.9, left=20, top=top, right=600, bottom=top + height))
        top += height + 20
    return lines


def test_unlabelled_name_needs_prominent_text(extractor: FieldExtractor) -> None:
    rows = [("ACME SYNTHETIC CORP", 30), ("JORDAN TESTER", 52), ("EMPLOYEE ID: EMP-004521", 30), ("DEPT: RESEARCH", 30)]
    result = extractor.extract(_lines(rows))
    assert result.full_name.value == "JORDAN TESTER"
    assert result.full_name.method == "unlabeled_prominent_line"

    same_size = [(text, 30) for text, _ in rows]
    assert extractor.extract(_lines(same_size)).full_name is None


def test_raw_text_and_confidence_are_preserved(extractor: FieldExtractor) -> None:
    line = OcrLine(text="NAME:  ALEX   SAMPLE", confidence=0.87, left=0, top=0, right=500, bottom=30)
    result = extractor.extract([line])
    assert result.full_name.value == "ALEX SAMPLE"
    assert result.full_name.raw_text == "NAME:  ALEX   SAMPLE"
    assert result.full_name.ocr_confidence == 0.87


def test_profiles_make_rules_extendable() -> None:
    labels = dict(DEFAULT_PROFILE.label_patterns)
    labels["full_name"] = labels["full_name"] + (r"nombre",)
    labels["date_of_birth"] = labels["date_of_birth"] + (r"fecha\s*de\s*nacimiento",)
    spanish = replace(DEFAULT_PROFILE, name="spanish_example", label_patterns=labels)
    result = FieldExtractor(profile=spanish, today=TODAY).extract_from_text(
        "NOMBRE: ALEX SAMPLE\nFECHA DE NACIMIENTO: 14/03/1991"
    )
    assert result.full_name.value == "ALEX SAMPLE"
    assert result.date_of_birth.value == "1991-03-14"
