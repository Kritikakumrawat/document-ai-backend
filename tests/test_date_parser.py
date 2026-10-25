"""Date parsing used for date of birth and expiry date."""

from datetime import date

import pytest

from app.services.date_parser import find_dates, fix_ocr_digits

TODAY = date(2026, 10, 22)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("14/03/1991", date(1991, 3, 14)),
        ("14-03-1991", date(1991, 3, 14)),
        ("14.03.1991", date(1991, 3, 14)),
        ("1991-03-14", date(1991, 3, 14)),
        ("1991/03/14", date(1991, 3, 14)),
        ("03/14/1991", date(1991, 3, 14)),  # 14 cannot be a month, so this is MDY
        ("14 MAR 1991", date(1991, 3, 14)),
        ("14-Mar-1991", date(1991, 3, 14)),
        ("14MAR1991", date(1991, 3, 14)),
        ("14 March 1991", date(1991, 3, 14)),
        ("March 14, 1991", date(1991, 3, 14)),
        ("MAR 14 1991", date(1991, 3, 14)),
        ("2nd Aug 1988", date(1988, 8, 2)),
        ("31 DEC 2027", date(2027, 12, 31)),
    ],
)
def test_common_formats(text: str, expected: date) -> None:
    matches = find_dates(f"DOB {text} SEX X", today=TODAY)
    assert [m.value for m in matches] == [expected]


def test_ambiguous_dates_follow_the_configured_order() -> None:
    (dmy,) = find_dates("04/05/1990", order="DMY", today=TODAY)
    (mdy,) = find_dates("04/05/1990", order="MDY", today=TODAY)
    assert (dmy.value, dmy.ambiguous) == (date(1990, 5, 4), True)
    assert (mdy.value, mdy.ambiguous) == (date(1990, 4, 5), True)
    assert find_dates("05/05/1990", today=TODAY)[0].ambiguous is False  # same either way


def test_two_digit_years() -> None:
    assert find_dates("14/03/91", today=TODAY)[0].value == date(1991, 3, 14)
    assert find_dates("14/03/31", today=TODAY)[0].value == date(2031, 3, 14)
    # For birth dates a two-digit year is moved into the past.
    assert find_dates("14/03/30", today=TODAY, prefer_past=True)[0].value == date(1930, 3, 14)


@pytest.mark.parametrize("text", ["31/02/2020", "45/45/2020", "00/00/0000", "14/13/1850"])
def test_impossible_dates_are_ignored(text: str) -> None:
    assert find_dates(text, today=TODAY) == []


@pytest.mark.parametrize("text", ["SYN-4821-7730", "123456789", "EMP-004521", "+1 555 0100"])
def test_numbers_that_are_not_dates(text: str) -> None:
    assert find_dates(text, today=TODAY) == []


def test_several_dates_are_returned_in_reading_order() -> None:
    matches = find_dates("ISS 15/03/2021 EXP 14 MAR 2031", today=TODAY)
    assert [m.value for m in matches] == [date(2021, 3, 15), date(2031, 3, 14)]
    assert [m.raw for m in matches] == ["15/03/2021", "14 MAR 2031"]


def test_ocr_letter_fixes_keep_positions() -> None:
    assert fix_ocr_digits("14/O3/l991") == "14/03/1991"
    assert fix_ocr_digits("OCT 14") == "OCT 14"  # month names are left alone
    assert len(fix_ocr_digits("1O/O1/2O20")) == len("1O/O1/2O20")
