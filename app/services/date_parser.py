"""Find and parse dates in OCR text.

Supported formats (separators ``/``, ``-`` or ``.``):

* ``14/03/1991``, ``03/14/1991`` and two-digit years such as ``14/03/91``
* ``1991-03-14`` (ISO style, year first)
* ``14 MAR 1991``, ``14-Mar-1991``, ``14MAR1991``
* ``MAR 14, 1991``, ``March 14 1991``

A numeric date such as ``04/05/1990`` could mean 4 May or April 5. When the
numbers themselves do not settle it (one of them above 12), the configured
order (``DMY`` or ``MDY``) is used and the match is flagged ``ambiguous``.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?![a-z])"
)

_ISO = re.compile(r"(?<!\d)(\d{4})([/\-.])(\d{1,2})\2(\d{1,2})(?!\d)")
_NUMERIC = re.compile(r"(?<!\d)(\d{1,2})\s?([/\-.])\s?(\d{1,2})\s?\2\s?(\d{4}|\d{2})(?!\d)")
_DAY_MONTH_YEAR = re.compile(
    r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?[\s\-/.,]*" + _MONTH + r"\.?[\s\-/.,]*(\d{4}|\d{2})(?!\d)",
    re.IGNORECASE,
)
_MONTH_DAY_YEAR = re.compile(
    r"(?<![a-z])" + _MONTH + r"\.?[\s\-/.]*(\d{1,2})(?:st|nd|rd|th)?,?[\s\-/.]*(\d{4})(?!\d)",
    re.IGNORECASE,
)

# Letters OCR often returns in place of digits, fixed only when they sit
# between digits or date separators (so month names stay untouched).
_DIGIT_LOOKALIKES = (
    (re.compile(r"(?<=[\d/\-.])[Oo](?=[\d/\-.])"), "0"),
    (re.compile(r"(?<=[\d/\-.])[Il|](?=[\d/\-.])"), "1"),
)


@dataclass(frozen=True)
class DateMatch:
    value: date
    raw: str
    start: int
    end: int
    ambiguous: bool = False


def fix_ocr_digits(text: str) -> str:
    """Replace look-alike letters inside numbers (``14/O3/1991`` -> ``14/03/1991``).

    Every replacement is one character for one character, so positions in the
    returned string match positions in the original text.
    """
    for pattern, digit in _DIGIT_LOOKALIKES:
        text = pattern.sub(digit, text)
    return text


def _expand_year(year_text: str, today: date) -> int:
    year = int(year_text)
    if len(year_text) == 4:
        return year
    pivot = (today.year + 20) % 100
    return 2000 + year if year <= pivot else 1900 + year


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    if not 1900 <= year <= 2100:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _move_to_past(value: date, year_digits: int, today: date) -> Optional[date]:
    """For birth dates written with two-digit years, '30' means 1930, not 2030."""
    if year_digits == 2 and value > today:
        try:
            return value.replace(year=value.year - 100)
        except ValueError:
            return None
    return value


def find_dates(
    text: str, order: str = "DMY", today: Optional[date] = None, prefer_past: bool = False
) -> List[DateMatch]:
    """Return every date found in ``text``, in reading order."""
    today = today or date.today()
    cleaned = fix_ocr_digits(text)
    taken = [False] * len(cleaned)
    found: List[DateMatch] = []

    for pattern in (_ISO, _DAY_MONTH_YEAR, _MONTH_DAY_YEAR, _NUMERIC):
        for match in pattern.finditer(cleaned):
            if any(taken[match.start():match.end()]):
                continue
            parsed = _parse_match(pattern, match, order, today, prefer_past)
            if parsed is None:
                continue
            value, ambiguous = parsed
            for i in range(match.start(), match.end()):
                taken[i] = True
            found.append(DateMatch(value, text[match.start():match.end()], match.start(), match.end(), ambiguous))

    return sorted(found, key=lambda item: item.start)


def _parse_match(
    pattern: "re.Pattern[str]", match: "re.Match[str]", order: str, today: date, prefer_past: bool
) -> Optional[Tuple[date, bool]]:
    ambiguous = False
    if pattern is _ISO:
        year_text, month, day = match.group(1), int(match.group(3)), int(match.group(4))
    elif pattern is _DAY_MONTH_YEAR:
        day, month, year_text = int(match.group(1)), _MONTHS[match.group(2)[:3].lower()], match.group(3)
    elif pattern is _MONTH_DAY_YEAR:
        month, day, year_text = _MONTHS[match.group(1)[:3].lower()], int(match.group(2)), match.group(3)
    else:
        first, second, year_text = int(match.group(1)), int(match.group(3)), match.group(4)
        if first > 12 >= second:
            day, month = first, second
        elif second > 12 >= first:
            month, day = first, second
        elif first <= 12 and second <= 12:
            ambiguous = first != second
            day, month = (first, second) if order == "DMY" else (second, first)
        else:
            return None

    value = _safe_date(_expand_year(year_text, today), month, day)
    if value is None:
        return None
    if prefer_past:
        value = _move_to_past(value, len(year_text), today)
        if value is None:
            return None
    return value, ambiguous
