"""Turn OCR lines into the four required fields.

The extractor is rule based and deliberately cautious:

1. **Labels first.** Text such as ``DOB``, ``Date of Birth``, ``Exp``,
   ``Valid until``, ``ID No`` or ``Name`` tells us which field a value belongs
   to. The value is read from the same line after the label or, if nothing
   useful follows the label, from the line directly below it.
2. **Validation.** Every candidate must look right for its field: a name is
   made of letters, a document number contains digits, a birth date lies in the
   past, an expiry date comes after the birth date, and so on.
3. **Careful fallbacks.** Without a label, a value is accepted only when the
   evidence is unambiguous (for example exactly one future date for the expiry
   date). If two candidates compete, the field stays ``None``; a number or date
   is never assigned just because it matches a pattern.

All label words and patterns live in an ``ExtractionProfile``. Supporting a new
document layout means adding a profile, not changing the algorithm.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

from app.services.date_parser import DateMatch, find_dates
from app.services.ocr_service import OcrLine

FIELD_NAMES = ("full_name", "document_number", "date_of_birth", "expiry_date")


@dataclass(frozen=True)
class ExtractionProfile:
    """Label vocabulary and patterns for one family of documents."""

    name: str
    # label kind -> regular expressions (matched case-insensitively, whole words)
    label_patterns: Dict[str, Tuple[str, ...]]
    # words that show a line is card text (titles, departments), not a person's name
    name_stopwords: FrozenSet[str]
    # (pattern name, regex) for document numbers that may appear without a label
    document_number_patterns: Tuple[Tuple[str, str], ...]


DEFAULT_PROFILE = ExtractionProfile(
    name="generic_id_card",
    label_patterns={
        "full_name": (
            r"full\s*name",
            r"(?:employee|holder|card\s*holder|member|student)'?s?\s*name",
            r"name",
        ),
        "surname": (r"sur\s*name", r"last\s*name", r"family\s*name"),
        "given_name": (r"given\s*names?", r"first\s*names?", r"fore\s*names?"),
        "document_number": (
            r"(?:id|identity|identification|card|document|doc|licen[cs]e|lic|dl|passport|badge"
            r"|employee|emp|staff|member|serial)\s*(?:no|number|num|nr)\.?",
            r"(?:employee|emp|staff|badge|member|student)\s*id",
            r"dln",
            r"(?:lic|id)\s*#",
            r"id(?=\s*[:#])",
        ),
        "date_of_birth": (
            r"date\s*of\s*birth",
            r"birth\s*date",
            r"d\s*\.?\s*[o0]\s*\.?\s*b\.?",
            r"born",
            r"birth",
        ),
        "expiry_date": (
            r"date\s*of\s*expiry",
            r"(?:expiry|expiration)\s*date",
            r"expir(?:y|es|ation)",
            r"exp\.?(?:\s*date)?",
            r"valid\s*(?:until|till|thru|through|to|up\s*to|upto)",
            r"good\s*(?:thru|through)",
            r"validity",
        ),
        "issue_date": (
            r"date\s*of\s*issue",
            r"issue\s*date",
            r"issued(?:\s*on)?",
            r"issue",
            r"iss\.?",
            r"doi",
            r"valid\s*from",
        ),
        # Labels we do not extract, listed so they end the value of the label before them.
        "other": (
            r"place\s*of\s*birth",
            r"sex", r"gender", r"height", r"hgt", r"weight", r"wgt", r"eyes", r"hair",
            r"class", r"address", r"addr", r"blood\s*(?:group|type)?",
            r"department", r"dept", r"designation", r"title", r"role",
            r"signature", r"nationality", r"restrictions?", r"endorsements?",
            r"phone", r"tel", r"e-?mail",
        ),
    },
    name_stopwords=frozenset(
        """
        DRIVER DRIVERS DRIVING LICENSE LICENCE PERMIT CARD IDENTITY IDENTIFICATION ID
        EMPLOYEE BADGE STAFF MEMBER STUDENT VISITOR CONTRACTOR NATIONAL GOVERNMENT STATE
        REPUBLIC DEPARTMENT MINISTRY TRANSPORT MOTOR VEHICLE VEHICLES AUTHORITY OFFICIAL
        COMPANY CORP CORPORATION INC LTD LLC LIMITED PVT GROUP TECHNOLOGIES TECHNOLOGY
        UNIVERSITY COLLEGE SCHOOL INSTITUTE HOSPITAL BANK SYNTHETIC TEST DATA NOT REAL
        VALID SIGNATURE HOLDER UNITED STATES KINGDOM OF THE AND FOR CLASS TYPE SECURITY
        ACCESS PHOTO ENGINEER ENGINEERING MANAGER DEVELOPER ANALYST OFFICER DIRECTOR
        INTERN HUMAN RESOURCES SALES MARKETING FINANCE OPERATIONS SOFTWARE QUALITY
        ASSURANCE RESEARCH SUPPORT ADMINISTRATION SERVICES DONOR ORGAN
        """.split()
    ),
    document_number_patterns=(
        ("hyphenated_alphanumeric", r"(?<![A-Z0-9])[A-Z]{1,4}-[A-Z0-9]{2,}(?:-[A-Z0-9]{2,})*(?![A-Z0-9])"),
        ("letter_prefixed_number", r"(?<![A-Z0-9])[A-Z]{1,3}\d{6,12}(?![A-Z0-9])"),
    ),
)


@dataclass
class FieldValue:
    value: str
    raw_text: str
    ocr_confidence: Optional[float]
    method: str


@dataclass
class ExtractionResult:
    full_name: Optional[FieldValue] = None
    document_number: Optional[FieldValue] = None
    date_of_birth: Optional[FieldValue] = None
    expiry_date: Optional[FieldValue] = None
    warnings: List[str] = field(default_factory=list)

    def found_count(self) -> int:
        return sum(getattr(self, name) is not None for name in FIELD_NAMES)


@dataclass
class _Line:
    text: str  # normalised
    raw: str  # as returned by the OCR engine
    confidence: Optional[float]
    left: int
    top: int
    right: int
    bottom: int

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)


@dataclass
class _LabelHit:
    kind: str
    line_index: int
    start: int
    end: int
    value: str  # text after the label, up to the next label on the line
    value_start: int


@dataclass
class _Candidate:
    """Text that may hold a field value, and where it came from."""

    text: str
    line_index: int
    offset: int  # position of ``text`` inside its line
    method: str  # "label" (same line) or "label_next_line"


_PUNCTUATION = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "“": '"', "”": '"', "„": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    }
)
_VALUE_EDGE = " \t:;.,-#|=>*_~"
_DIGIT_FIXES = (
    (re.compile(r"(?<=\d)[OQ](?=\d)"), "0"),
    (re.compile(r"(?<=\d)[IL|](?=\d)"), "1"),
    (re.compile(r"(?<=\d)S(?=\d)"), "5"),
    (re.compile(r"(?<=\d)B(?=\d)"), "8"),
    (re.compile(r"(?<=\d)Z(?=\d)"), "2"),
)


def normalize_text(text: str) -> str:
    """Unicode-normalise, unify quotes and dashes, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION)
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()


class FieldExtractor:
    def __init__(
        self,
        profile: ExtractionProfile = DEFAULT_PROFILE,
        date_order: str = "DMY",
        today: Optional[date] = None,
    ) -> None:
        self.profile = profile
        self.date_order = date_order
        self._today = today
        self._label_regexes = [
            (kind, re.compile(r"(?<![A-Za-z0-9])(?:" + pattern + r")(?![A-Za-z])", re.IGNORECASE))
            for kind, patterns in profile.label_patterns.items()
            for pattern in patterns
        ]
        self._id_regexes = [(name, re.compile(regex)) for name, regex in profile.document_number_patterns]

    @property
    def today(self) -> date:
        return self._today or date.today()

    # --- public API --------------------------------------------------------
    def extract_from_text(self, text: str) -> ExtractionResult:
        """Convenience wrapper for plain text: each line is placed below the previous one."""
        lines = [
            OcrLine(text=line, confidence=None, left=0, top=index * 40, right=1000, bottom=index * 40 + 30)
            for index, line in enumerate(line for line in text.splitlines() if line.strip())
        ]
        return self.extract(lines)

    def extract(self, ocr_lines: Sequence[OcrLine]) -> ExtractionResult:
        lines = [
            _Line(normalize_text(line.text), line.text, line.confidence, line.left, line.top, line.right, line.bottom)
            for line in ocr_lines
            if line.text and line.text.strip()
        ]
        hits = [self._label_hits(index, line) for index, line in enumerate(lines)]
        result = ExtractionResult()
        result.full_name = self._extract_name(lines, hits, result.warnings)
        result.document_number = self._extract_document_number(lines, hits, result.warnings)
        result.date_of_birth, result.expiry_date = self._extract_dates(lines, hits, result.warnings)
        return result

    # --- labels ------------------------------------------------------------
    def _label_hits(self, line_index: int, line: _Line) -> List[_LabelHit]:
        found = []
        for kind, regex in self._label_regexes:
            for match in regex.finditer(line.text):
                found.append((match.start(), match.end(), kind))
        # Keep non-overlapping labels: earliest first, and the longest at the same start
        # (so "DATE OF BIRTH" wins over "BIRTH", "FULL NAME" over "NAME").
        found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        chosen: List[Tuple[int, int, str]] = []
        for start, end, kind in found:
            if not chosen or start >= chosen[-1][1]:
                chosen.append((start, end, kind))

        hits = []
        for i, (start, end, kind) in enumerate(chosen):
            stop = chosen[i + 1][0] if i + 1 < len(chosen) else len(line.text)
            segment = line.text[end:stop]
            stripped = segment.lstrip(_VALUE_EDGE)
            value_start = end + (len(segment) - len(stripped))
            hits.append(_LabelHit(kind, line_index, start, end, stripped.rstrip(_VALUE_EDGE), value_start))
        return hits

    def _value_below(self, lines: List[_Line], hits: List[List[_LabelHit]], hit: _LabelHit) -> Optional[_Candidate]:
        """Find the text written under a label (for layouts with the label above the value)."""
        line = lines[hit.line_index]
        width = max(1, len(line.text))
        label_left = line.left + (line.right - line.left) * hit.start / width
        label_right = line.left + (line.right - line.left) * hit.end / width
        window_left, window_right = label_left - line.height, label_right + 6 * line.height

        best: Optional[int] = None
        for index, other in enumerate(lines):
            below = other.top >= line.top + 0.5 * line.height and other.top - line.bottom <= 2 * line.height
            overlaps = other.left <= window_right and other.right >= window_left
            if index != hit.line_index and below and overlaps:
                if best is None or other.top < lines[best].top:
                    best = index
        if best is None:
            return None
        below_hits = hits[best]
        text = lines[best].text if not below_hits else lines[best].text[: below_hits[0].start]
        stripped = text.lstrip(_VALUE_EDGE)
        offset = len(text) - len(stripped)
        stripped = stripped.rstrip(_VALUE_EDGE)
        return _Candidate(stripped, best, offset, "label_next_line") if stripped else None

    def _candidates_per_label(
        self, lines: List[_Line], hits: List[List[_LabelHit]], kind: str
    ) -> Iterator[List[_Candidate]]:
        """For each label of ``kind``: the text after it, then the text below it."""
        for hit in (h for line_hits in hits for h in line_hits if h.kind == kind):
            options = []
            if hit.value:
                options.append(_Candidate(hit.value, hit.line_index, hit.value_start, "label"))
            below = self._value_below(lines, hits, hit)
            if below is not None:
                options.append(below)
            yield options

    def _field_value(self, value: str, line: _Line, method: str) -> FieldValue:
        return FieldValue(value=value, raw_text=line.raw, ocr_confidence=line.confidence, method=method)

    def _first_valid(
        self,
        lines: List[_Line],
        hits: List[List[_LabelHit]],
        kind: str,
        clean: Callable[[str], Optional[str]],
    ) -> Optional[FieldValue]:
        for options in self._candidates_per_label(lines, hits, kind):
            for candidate in options:
                value = clean(candidate.text)
                if value:
                    return self._field_value(value, lines[candidate.line_index], candidate.method)
        return None

    # --- full name -----------------------------------------------------------
    def _extract_name(
        self, lines: List[_Line], hits: List[List[_LabelHit]], warnings: List[str]
    ) -> Optional[FieldValue]:
        full = self._first_valid(lines, hits, "full_name", self._clean_name)
        if full:
            return full

        given = self._first_valid(lines, hits, "given_name", self._clean_name)
        surname = self._first_valid(lines, hits, "surname", self._clean_name)
        if given and surname:
            confidences = [c for c in (given.ocr_confidence, surname.ocr_confidence) if c is not None]
            return FieldValue(
                value=f"{given.value} {surname.value}",
                raw_text=f"{given.raw_text} | {surname.raw_text}",
                ocr_confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
                method="label_given_name_and_surname",
            )
        if given or surname:
            part = given or surname
            which = "given name" if given else "surname"
            warnings.append(f"Only the {which} was found; full_name may be incomplete.")
            part.method = f"label_{which.replace(' ', '_')}_only"
            return part

        return self._unlabeled_name(lines, hits, warnings)

    def _clean_name(self, text: str) -> Optional[str]:
        digit = re.search(r"\d", text)
        if digit:
            text = text[: digit.start()]  # e.g. "ALEX SAMPLE 4d" -> "ALEX SAMPLE"
        text = re.sub(r"[^\w\s'\-.,]|_", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" .,-'")
        words = text.split()
        if not 1 <= len(words) <= 6 or sum(ch.isalpha() for ch in text) < 2:
            return None
        if any(len(word) > 30 for word in words):
            return None
        if not all(ch.isalpha() or ch in " '-.," for ch in text):
            return None
        stopwords = sum(word.upper().strip(".,'-") in self.profile.name_stopwords for word in words)
        if stopwords * 2 >= len(words):
            return None
        return text

    def _unlabeled_name(
        self, lines: List[_Line], hits: List[List[_LabelHit]], warnings: List[str]
    ) -> Optional[FieldValue]:
        """Badges often print the name without a label, in a larger font than the rest.

        A line qualifies only if it is clearly taller than the typical text line,
        has 2-4 purely alphabetic words, contains no card vocabulary and was read
        with reasonable confidence. Exactly one such line must exist.
        """
        text_heights = sorted(line.height for line in lines if sum(ch.isalnum() for ch in line.text) >= 2)
        if not text_heights:
            return None
        typical_height = text_heights[len(text_heights) // 2]

        candidates = []
        for index, line in enumerate(lines):
            if hits[index] or line.height < 1.25 * typical_height:
                continue
            words = line.text.split()
            if not 2 <= len(words) <= 4:
                continue
            if not all(re.fullmatch(r"[^\W\d_][^\W\d_'\-.]*", word) for word in words):
                continue
            if sum(len(word) >= 2 for word in words) < 2 or sum(ch.isalpha() for ch in line.text) < 5:
                continue
            if any(word.upper().strip(".,'-") in self.profile.name_stopwords for word in words):
                continue
            if line.confidence is not None and line.confidence < 0.6:
                continue
            candidates.append(index)

        if len(candidates) == 1:
            line = lines[candidates[0]]
            warnings.append("full_name was found without a label (largest text line); please verify it.")
            return self._field_value(line.text, line, "unlabeled_prominent_line")
        if len(candidates) > 1:
            warnings.append("No name label was found and several lines could be a name; full_name left empty.")
        return None

    # --- document number -----------------------------------------------------
    def _extract_document_number(
        self, lines: List[_Line], hits: List[List[_LabelHit]], warnings: List[str]
    ) -> Optional[FieldValue]:
        labeled = self._first_valid(lines, hits, "document_number", self._clean_document_number)
        if labeled:
            return labeled

        found: Dict[str, Tuple[int, str]] = {}
        for index, line in enumerate(lines):
            upper = line.text.upper()
            for name, regex in self._id_regexes:
                for match in regex.finditer(upper):
                    token = match.group(0)
                    if sum(ch.isdigit() for ch in token) < 5 or find_dates(token, self.date_order, self.today):
                        continue
                    found.setdefault(token, (index, name))
        if len(found) == 1:
            token, (index, name) = next(iter(found.items()))
            return self._field_value(token, lines[index], f"unlabeled_pattern:{name}")
        if len(found) > 1:
            warnings.append("Several unlabelled values look like document numbers; document_number left empty.")
        return None

    def _clean_document_number(self, text: str) -> Optional[str]:
        tokens = text.upper().split()
        picked: List[str] = []
        for i, token in enumerate(tokens):
            token = token.strip(".,;:|")
            if not token or not re.fullmatch(r"[A-Z0-9](?:[A-Z0-9\-/]*[A-Z0-9])?", token):
                break
            if any(ch.isdigit() for ch in token):
                for pattern, digit in _DIGIT_FIXES:
                    token = pattern.sub(digit, token)
                picked.append(token)
            elif not picked and len(token) <= 4 and i + 1 < len(tokens) and any(c.isdigit() for c in tokens[i + 1]):
                picked.append(token)  # short prefix such as "SYN 4821 7730"
            else:
                break
        candidate = " ".join(picked)
        compact = candidate.replace(" ", "")
        if sum(ch.isdigit() for ch in compact) < 3 or not 4 <= len(compact) <= 30:
            return None
        if find_dates(candidate, self.date_order, self.today):
            return None  # a date is not a document number
        return candidate

    # --- dates -----------------------------------------------------------------
    def _dates_for_label(
        self, lines: List[_Line], hits: List[List[_LabelHit]], kind: str, prefer_past: bool
    ) -> List[Tuple[List[DateMatch], int, int, str]]:
        """Dates found for each label of ``kind``: (dates, line index, offset, method).

        The line below a label is only read when no date follows the label itself.
        """
        results = []
        for options in self._candidates_per_label(lines, hits, kind):
            for candidate in options:
                dates = find_dates(candidate.text, self.date_order, self.today, prefer_past=prefer_past)
                if dates:
                    results.append((dates, candidate.line_index, candidate.offset, candidate.method))
                    break
        return results

    def _date_field(self, match: DateMatch, line: _Line, method: str, label: str, warnings: List[str]) -> FieldValue:
        if match.ambiguous:
            warnings.append(
                f"{label} '{match.raw}' could be day/month or month/day; read as {self.date_order}."
            )
        return self._field_value(match.value.isoformat(), line, method)

    def _extract_dates(
        self, lines: List[_Line], hits: List[List[_LabelHit]], warnings: List[str]
    ) -> Tuple[Optional[FieldValue], Optional[FieldValue]]:
        today = self.today
        claimed = set()  # (line index, start) of dates that belong to a labelled field

        birth: Optional[Tuple[DateMatch, int, str]] = None
        for dates, line_index, offset, method in self._dates_for_label(lines, hits, "date_of_birth", True):
            claimed.update((line_index, offset + d.start) for d in dates)
            candidate = dates[0]
            if birth is None and self._plausible_birth(candidate.value, today):
                birth = (candidate, line_index, method)
            elif birth is None:
                warnings.append(f"Date of birth '{candidate.raw}' is not plausible and was ignored.")

        expiry: Optional[Tuple[DateMatch, int, str]] = None
        for dates, line_index, offset, method in self._dates_for_label(lines, hits, "expiry_date", False):
            claimed.update((line_index, offset + d.start) for d in dates)
            candidate = max(dates, key=lambda d: d.value)  # "valid 2021-2031" -> the later date
            if expiry is None:
                expiry = (candidate, line_index, method)

        for dates, line_index, offset, _ in self._dates_for_label(lines, hits, "issue_date", False):
            claimed.update((line_index, offset + d.start) for d in dates)

        # Unlabelled dates are used only when the evidence is unambiguous.
        pool = [
            (d, index)
            for index, line in enumerate(lines)
            for d in find_dates(line.text, self.date_order, today)
            if (index, d.start) not in claimed
        ]
        if birth is None and pool:
            birth = self._unlabeled_birth(pool, today)
        if expiry is None and pool:
            future = [(d, index) for d, index in pool if d.value > today]
            if len({d.value for d, _ in future}) == 1:
                expiry = (future[0][0], future[0][1], "unlabeled_only_future_date")

        if birth and expiry and expiry[0].value <= birth[0].value:
            warnings.append("The expiry date is not after the date of birth, so it was discarded.")
            expiry = None

        birth_field = expiry_field = None
        if birth:
            birth_field = self._date_field(birth[0], lines[birth[1]], birth[2], "Date of birth", warnings)
        if expiry:
            expiry_field = self._date_field(expiry[0], lines[expiry[1]], expiry[2], "Expiry date", warnings)
        return birth_field, expiry_field

    @staticmethod
    def _plausible_birth(value: date, today: date) -> bool:
        return value < today and today.year - value.year <= 120

    @staticmethod
    def _unlabeled_birth(pool: List[Tuple[DateMatch, int]], today: date) -> Optional[Tuple[DateMatch, int, str]]:
        """Pick the birth date from unlabelled dates only when one clearly stands out.

        The oldest date must be at least ten years older than every other date on
        the card (issue and expiry dates are much more recent) and imply an age of
        10-110 years. If it is the only date on the card, the age must be 18-110,
        because a lone recent date is more likely an issue date.
        """
        oldest, index = min(pool, key=lambda item: item[0].value)
        age = today.year - oldest.value.year
        others = [d.value for d, _ in pool if d.value != oldest.value]
        if not others:
            return (oldest, index, "unlabeled_only_past_date") if 18 <= age <= 110 else None
        if 10 <= age <= 110 and all(other.year - oldest.value.year >= 10 for other in others):
            return oldest, index, "unlabeled_oldest_date"
        return None
