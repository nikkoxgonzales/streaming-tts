"""
Text normalization module for TTS processing.
Handles various text formats including URLs, emails, numbers, money, and special characters.
Converts them into a format suitable for text-to-speech processing.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import inflect
    INFLECT_ENGINE = inflect.engine()
    HAS_INFLECT = True
except ImportError:
    INFLECT_ENGINE = None
    HAS_INFLECT = False


@dataclass
class NormalizationOptions:
    """Options for text normalization."""
    normalize: bool = True
    url_normalization: bool = True
    email_normalization: bool = True
    phone_normalization: bool = True
    unit_normalization: bool = True
    optional_pluralization_normalization: bool = True
    replace_remaining_symbols: bool = True


# Constants
VALID_TLDS = [
    "com", "org", "net", "edu", "gov", "mil", "int", "biz", "info", "name",
    "pro", "io", "co", "ai", "app", "dev", "xyz", "me", "tv", "uk", "us",
    "ca", "de", "fr", "jp", "cn", "ru", "br", "au", "in", "it", "nl", "es",
]

VALID_UNITS = {
    "m": "meter", "cm": "centimeter", "mm": "millimeter", "km": "kilometer",
    "in": "inch", "ft": "foot", "yd": "yard", "mi": "mile",
    "g": "gram", "kg": "kilogram", "mg": "milligram",
    "s": "second", "ms": "millisecond", "min": "minutes", "h": "hour",
    "l": "liter", "ml": "mililiter",
    "kph": "kilometer per hour", "mph": "mile per hour",
    "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
    "kb": "kilobyte", "mb": "megabyte", "gb": "gigabyte", "tb": "terabyte",
    "kbps": "kilobits per second", "mbps": "megabits per second",
    "px": "pixel",
}

SYMBOL_REPLACEMENTS = {
    '~': ' ', '@': ' at ', '#': ' number ', '$': ' dollar ', '%': ' percent ',
    '^': ' ', '&': ' and ', '*': ' ', '_': ' ', '|': ' ', '\\': ' ',
    '/': ' slash ', '=': ' equals ', '+': ' plus ',
}

MONEY_UNITS = {
    "$": ("dollar", "cent"),
    "\u00a3": ("pound", "pence"),  # £
    "\u20ac": ("euro", "cent"),    # €
}

# Pre-compiled regex patterns
EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE
)

_URL_TLD_PATTERN = "|".join(VALID_TLDS)
URL_PATTERN = re.compile(
    rf"(https?://|www\.|)+(localhost|[a-zA-Z0-9.-]+(\.(?:{_URL_TLD_PATTERN}))+|[0-9]{{1,3}}\.[0-9]{{1,3}}\.[0-9]{{1,3}}\.[0-9]{{1,3}})(:[0-9]+)?([/?][^\s]*)?",
    re.IGNORECASE,
)

_UNIT_NAMES_PATTERN = "|".join(sorted(list(VALID_UNITS.keys()), reverse=True))
UNIT_PATTERN = re.compile(
    rf"((?<!\w)([+-]?)(\d{{1,3}}(,\d{{3}})*|\d+)(\.\d+)?)\s*({_UNIT_NAMES_PATTERN}){{1}}(?=[^\w\d]{{1}}|\b)",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"([0-9]{1,2} ?: ?[0-9]{2}( ?: ?[0-9]{2})?)( ?(pm|am)\b)?", re.IGNORECASE
)

_MONEY_SYMBOLS = "".join(MONEY_UNITS.keys())
MONEY_PATTERN = re.compile(
    rf"(-?)([{_MONEY_SYMBOLS}])(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b|t)*)\b",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"(-?)(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b)*)\b",
    re.IGNORECASE,
)


def _number_to_words(num: float) -> str:
    """Convert a number to words using inflect if available, else basic conversion."""
    if HAS_INFLECT and INFLECT_ENGINE:
        return INFLECT_ENGINE.number_to_words(num)
    # Basic fallback for common numbers
    return str(num)


def _plural(word: str, count: float) -> str:
    """Pluralize a word if needed."""
    if HAS_INFLECT and INFLECT_ENGINE:
        return INFLECT_ENGINE.plural(word, count)
    return word if abs(count) == 1 else word + "s"


def _no(word: str, count) -> str:
    """Return 'count word(s)' with proper pluralization."""
    if HAS_INFLECT and INFLECT_ENGINE:
        return INFLECT_ENGINE.no(word, count)
    return f"{count} {_plural(word, float(count) if isinstance(count, str) else count)}"


def _conditional_int(number: float, threshold: float = 0.00001) -> int | float:
    """Convert to int if close to a whole number."""
    if abs(round(number) - number) < threshold:
        return int(round(number))
    return number


def _translate_multiplier(multiplier: str) -> str:
    """Translate multiplier abbreviations to words."""
    translation = {"k": "thousand", "m": "million", "b": "billion", "t": "trillion"}
    return translation.get(multiplier.lower(), multiplier.strip())


def _split_four_digit(number: float) -> str:
    """Split four-digit years into spoken form (e.g., 1984 -> nineteen eighty-four)."""
    num_str = str(_conditional_int(number))
    part1, part2 = num_str[:2], num_str[2:]
    return f"{_number_to_words(int(part1))} {_number_to_words(int(part2))}"


def _handle_units(match: re.Match) -> str:
    """Convert units to their full form."""
    unit_string = match.group(6).strip()
    if unit_string.lower() in VALID_UNITS:
        unit_parts = VALID_UNITS[unit_string.lower()].split(" ")
        # Handle byte vs bit
        if unit_parts[0].endswith("bit"):
            b_case = unit_string[min(1, len(unit_string) - 1)]
            if b_case == "B":
                unit_parts[0] = unit_parts[0][:-3] + "byte"
        number = match.group(1).strip()
        unit_parts[0] = _no(unit_parts[0], number)
        return " ".join(unit_parts)
    return match.group(0)


def _handle_numbers(match: re.Match) -> str:
    """Convert numbers to spoken form."""
    number_str = match.group(2)
    try:
        number = float(number_str)
    except ValueError:
        return match.group()

    if match.group(1) == "-":
        number *= -1

    multiplier = _translate_multiplier(match.group(3))
    number = _conditional_int(number)

    if multiplier:
        return f"{_number_to_words(number)} {multiplier}"

    # Handle four-digit years (1500-2099 that aren't divisible by 100)
    if (isinstance(number, int) and len(str(abs(number))) == 4
        and 1500 < abs(number) < 2100 and number % 100 != 0):
        return _split_four_digit(number)

    return _number_to_words(number)


def _handle_money(match: re.Match) -> str:
    """Convert money expressions to spoken form."""
    symbol = match.group(2)
    if symbol not in MONEY_UNITS:
        return match.group()

    bill, coin = MONEY_UNITS[symbol]
    number_str = match.group(3)

    try:
        number = float(number_str)
    except ValueError:
        return match.group()

    if match.group(1) == "-":
        number *= -1

    multiplier = _translate_multiplier(match.group(4))

    if multiplier:
        return f"{_number_to_words(_conditional_int(number))} {multiplier} {_plural(bill, number)}"

    if number % 1 == 0:
        return f"{_number_to_words(_conditional_int(number))} {_plural(bill, number)}"

    # Handle cents
    sub_number = int(str(number).split(".")[-1].ljust(2, "0"))
    return (f"{_number_to_words(int(math.floor(number)))} {_plural(bill, number)} "
            f"and {_number_to_words(sub_number)} {_plural(coin, sub_number)}")


def _handle_decimal(match: re.Match) -> str:
    """Convert decimal numbers to spoken form."""
    parts = match.group().split(".")
    return f"{parts[0]} point " + " ".join(parts[1])


def _handle_email(match: re.Match) -> str:
    """Convert email addresses to speakable format."""
    email = match.group(0)
    parts = email.split("@")
    if len(parts) == 2:
        user, domain = parts
        domain = domain.replace(".", " dot ")
        return f"{user} at {domain}"
    return email


def _handle_url(match: re.Match) -> str:
    """Make URLs speakable."""
    if not match:
        return ""

    url = match.group(0).strip()

    # Handle protocol
    url = re.sub(r"^https?://", lambda m: "https " if "https" in m.group() else "http ", url, flags=re.IGNORECASE)
    url = re.sub(r"^www\.", "www ", url, flags=re.IGNORECASE)

    # Handle port numbers
    url = re.sub(r":(\d+)(?=/|$)", lambda m: f" colon {m.group(1)}", url)

    # Split domain and path
    parts = url.split("/", 1)
    domain = parts[0].replace(".", " dot ")
    path = parts[1] if len(parts) > 1 else ""

    if path:
        url = f"{domain} slash {path}"
    else:
        url = domain

    # Replace symbols
    replacements = [("-", " dash "), ("_", " underscore "), ("?", " question mark "),
                    ("=", " equals "), ("&", " and "), ("%", " percent "), ("/", " slash ")]
    for old, new in replacements:
        url = url.replace(old, new)

    return re.sub(r"\s+", " ", url).strip()


def _handle_time(match: re.Match) -> str:
    """Convert time expressions to spoken form."""
    groups = match.groups()
    time_parts = groups[0].split(":")

    numbers = [_number_to_words(int(time_parts[0].strip()))]

    minute = int(time_parts[1].strip())
    if minute == 0:
        pass  # Will add "o'clock" later if no am/pm
    elif minute < 10:
        numbers.append(f"oh {_number_to_words(minute)}")
    else:
        numbers.append(_number_to_words(minute))

    # Handle seconds if present
    if len(time_parts) > 2:
        seconds = int(time_parts[2].strip())
        numbers.append(f"and {_number_to_words(seconds)} {_plural('second', seconds)}")
    elif groups[2]:  # am/pm
        numbers.append(groups[2].strip())
    elif minute == 0:
        numbers.append("o'clock")

    return " ".join(numbers)


def _handle_phone_number(match: re.Match) -> str:
    """Convert phone numbers to spoken form."""
    groups = list(match.groups())
    parts = []

    # Country code
    if groups[0]:
        parts.append(_number_to_words(int(groups[0].replace("+", ""))))

    # Area code
    if groups[2]:
        area = groups[2].replace("(", "").replace(")", "")
        parts.append(" ".join(_number_to_words(int(d)) for d in area))

    # Rest of number
    if groups[3]:
        parts.append(" ".join(_number_to_words(int(d)) for d in groups[3]))
    if groups[4]:
        parts.append(" ".join(_number_to_words(int(d)) for d in groups[4]))

    return ", ".join(parts)


def normalize_text(
    text: str,
    options: Optional[NormalizationOptions] = None
) -> str:
    """
    Normalize text for TTS processing.

    Args:
        text: The text to normalize
        options: Normalization options (uses defaults if None)

    Returns:
        Normalized text suitable for TTS
    """
    if options is None:
        options = NormalizationOptions()

    if not options.normalize:
        return text

    # Handle emails first
    if options.email_normalization:
        text = EMAIL_PATTERN.sub(_handle_email, text)

    # Handle URLs
    if options.url_normalization:
        text = URL_PATTERN.sub(_handle_url, text)

    # Handle units
    if options.unit_normalization:
        text = UNIT_PATTERN.sub(_handle_units, text)

    # Handle optional pluralization (s)
    if options.optional_pluralization_normalization:
        text = re.sub(r"\(s\)", "s", text)

    # Handle phone numbers
    if options.phone_normalization:
        text = re.sub(
            r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})",
            _handle_phone_number,
            text,
        )

    # Replace quotes and brackets
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')

    # Handle CJK punctuation
    for a, b in zip("\u3001\u3002\uff01\uff0c\uff1a\uff1b\uff1f\u2013", ",.!,:;?-"):
        text = text.replace(a, b + " ")

    # Handle time
    text = TIME_PATTERN.sub(_handle_time, text)

    # Clean whitespace
    text = re.sub(r"[^\S \n]", " ", text)
    text = re.sub(r"  +", " ", text)
    text = text.replace('\n', ' ').replace('\r', ' ')

    # Handle titles
    text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
    text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
    text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
    text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)

    # Handle numbers and money
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # Remove commas in numbers
    text = MONEY_PATTERN.sub(_handle_money, text)
    text = NUMBER_PATTERN.sub(_handle_numbers, text)
    text = re.sub(r"\d*\.\d+", _handle_decimal, text)

    # Replace remaining symbols
    if options.replace_remaining_symbols:
        for symbol, replacement in SYMBOL_REPLACEMENTS.items():
            text = text.replace(symbol, replacement)

    # Final cleanup
    text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()
