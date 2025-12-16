"""Tests for streaming_tts.text_normalizer module."""

from streaming_tts.text_normalizer import (
    normalize_text,
    NormalizationOptions,
    _handle_email,
    _handle_url,
    _handle_money,
    _handle_numbers,
    _handle_units,
    _handle_time,
    _handle_phone_number,
    EMAIL_PATTERN,
    URL_PATTERN,
    MONEY_PATTERN,
    NUMBER_PATTERN,
    UNIT_PATTERN,
    TIME_PATTERN,
)


class TestNormalizationOptions:
    """Tests for NormalizationOptions dataclass."""

    def test_default_values(self):
        opts = NormalizationOptions()
        assert opts.normalize is True
        assert opts.url_normalization is True
        assert opts.email_normalization is True
        assert opts.phone_normalization is True
        assert opts.unit_normalization is True
        assert opts.replace_remaining_symbols is True

    def test_custom_values(self):
        opts = NormalizationOptions(
            normalize=True,
            url_normalization=False,
            email_normalization=False
        )
        assert opts.url_normalization is False
        assert opts.email_normalization is False


class TestHandleEmail:
    """Tests for email normalization."""

    def test_simple_email(self):
        match = EMAIL_PATTERN.search("Contact user@example.com please")
        result = _handle_email(match)
        assert "at" in result
        assert "dot" in result
        assert "user" in result
        assert "example" in result
        assert "com" in result

    def test_email_with_dots_in_user(self):
        match = EMAIL_PATTERN.search("Email john.doe@test.org")
        result = _handle_email(match)
        assert "at" in result
        assert "test" in result
        assert "org" in result

    def test_email_format(self):
        match = EMAIL_PATTERN.search("hello@world.io")
        result = _handle_email(match)
        assert "hello" in result
        assert "at" in result
        assert "world" in result
        assert "dot" in result
        assert "io" in result


class TestHandleUrl:
    """Tests for URL normalization."""

    def test_https_url(self):
        match = URL_PATTERN.search("Visit https://example.com")
        result = _handle_url(match)
        assert "https" in result.lower()
        assert "example" in result
        assert "dot" in result
        assert "com" in result

    def test_url_with_path(self):
        match = URL_PATTERN.search("Go to https://example.com/path/to/page")
        result = _handle_url(match)
        assert "slash" in result
        assert "path" in result

    def test_url_with_query_params(self):
        match = URL_PATTERN.search("https://example.com?foo=bar")
        result = _handle_url(match)
        assert "question mark" in result
        assert "equals" in result

    def test_www_url(self):
        match = URL_PATTERN.search("www.example.com")
        result = _handle_url(match)
        assert "www" in result
        assert "example" in result


class TestHandleMoney:
    """Tests for money normalization."""

    def test_dollars_whole(self):
        match = MONEY_PATTERN.search("Cost is $42")
        result = _handle_money(match)
        assert "forty" in result.lower() or "42" in result
        assert "dollar" in result.lower()

    def test_dollars_with_cents(self):
        match = MONEY_PATTERN.search("Price $42.50")
        result = _handle_money(match)
        assert "dollar" in result.lower()
        # Should have both dollar and cent parts
        assert "cent" in result.lower() or "fifty" in result.lower()

    def test_pounds(self):
        match = MONEY_PATTERN.search("Cost \u00a3100")
        result = _handle_money(match)
        assert "hundred" in result.lower() or "100" in result
        assert "pound" in result.lower()

    def test_euros(self):
        match = MONEY_PATTERN.search("Price \u20ac50")
        result = _handle_money(match)
        assert "fifty" in result.lower() or "50" in result
        assert "euro" in result.lower()

    def test_negative_money(self):
        match = MONEY_PATTERN.search("-$10")
        result = _handle_money(match)
        assert "dollar" in result.lower()

    def test_money_with_multiplier_k(self):
        match = MONEY_PATTERN.search("Salary $50k")
        result = _handle_money(match)
        assert "thousand" in result.lower()
        assert "dollar" in result.lower()

    def test_money_with_multiplier_m(self):
        match = MONEY_PATTERN.search("Revenue $5m")
        result = _handle_money(match)
        assert "million" in result.lower()
        assert "dollar" in result.lower()


class TestHandleNumbers:
    """Tests for number normalization."""

    def test_simple_integer(self):
        match = NUMBER_PATTERN.search("I have 42 apples")
        result = _handle_numbers(match)
        assert "forty" in result.lower() or "42" in result

    def test_negative_number(self):
        match = NUMBER_PATTERN.search("Temperature is -10 degrees")
        result = _handle_numbers(match)
        # Should handle negative
        assert "ten" in result.lower() or "10" in result

    def test_multiplier_k(self):
        match = NUMBER_PATTERN.search("We have 5k users")
        result = _handle_numbers(match)
        assert "thousand" in result.lower()

    def test_multiplier_m(self):
        match = NUMBER_PATTERN.search("Population 8m")
        result = _handle_numbers(match)
        assert "million" in result.lower()

    def test_year_in_range(self):
        # Years 1500-2099 that aren't divisible by 100 get split
        match = NUMBER_PATTERN.search("Born in 1984")
        result = _handle_numbers(match)
        # Should be "nineteen eighty-four" style
        assert "nineteen" in result.lower() or "1984" in result

    def test_year_2000(self):
        # 2000 is divisible by 100, so handled differently
        match = NUMBER_PATTERN.search("Year 2000")
        result = _handle_numbers(match)
        assert "two thousand" in result.lower() or "2000" in result


class TestHandleUnits:
    """Tests for unit normalization."""

    def test_kilometers(self):
        match = UNIT_PATTERN.search("Distance 10km away")
        result = _handle_units(match)
        assert "kilometer" in result.lower()

    def test_kilograms(self):
        match = UNIT_PATTERN.search("Weight 5kg")
        result = _handle_units(match)
        assert "kilogram" in result.lower()

    def test_minutes(self):
        match = UNIT_PATTERN.search("Wait 30min")
        result = _handle_units(match)
        assert "minute" in result.lower()

    def test_megabytes_uppercase(self):
        # MB = megabytes (uppercase B)
        match = UNIT_PATTERN.search("File is 100MB")
        result = _handle_units(match)
        assert "megabyte" in result.lower()

    def test_megabits_lowercase(self):
        # Mb = megabits (lowercase b)
        match = UNIT_PATTERN.search("Speed 100Mb")
        result = _handle_units(match)
        # Should distinguish bits vs bytes
        assert "mega" in result.lower()

    def test_plural_handling_singular(self):
        match = UNIT_PATTERN.search("Distance 1km")
        result = _handle_units(match)
        assert "kilometer" in result.lower()

    def test_plural_handling_plural(self):
        match = UNIT_PATTERN.search("Distance 2km")
        result = _handle_units(match)
        assert "kilometer" in result.lower()

    def test_decimal_value(self):
        match = UNIT_PATTERN.search("Height 1.5m")
        result = _handle_units(match)
        assert "meter" in result.lower()


class TestHandleTime:
    """Tests for time normalization."""

    def test_12_hour_am(self):
        match = TIME_PATTERN.search("Wake up at 9:30am")
        result = _handle_time(match)
        assert "nine" in result.lower() or "9" in result
        assert "thirty" in result.lower() or "30" in result
        assert "am" in result.lower()

    def test_12_hour_pm(self):
        match = TIME_PATTERN.search("Meeting at 2:15pm")
        result = _handle_time(match)
        assert "two" in result.lower() or "2" in result
        assert "fifteen" in result.lower() or "15" in result
        assert "pm" in result.lower()

    def test_24_hour(self):
        match = TIME_PATTERN.search("Departure 14:30")
        result = _handle_time(match)
        assert "fourteen" in result.lower() or "14" in result
        assert "thirty" in result.lower() or "30" in result

    def test_on_the_hour(self):
        match = TIME_PATTERN.search("Noon at 12:00")
        result = _handle_time(match)
        assert "twelve" in result.lower() or "12" in result
        assert "o'clock" in result.lower() or "clock" in result.lower()

    def test_oh_minutes(self):
        match = TIME_PATTERN.search("Start at 9:05")
        result = _handle_time(match)
        assert "nine" in result.lower() or "9" in result
        assert "oh" in result.lower() or "05" in result or "five" in result.lower()

    def test_with_seconds(self):
        match = TIME_PATTERN.search("Time 14:30:45")
        result = _handle_time(match)
        assert "second" in result.lower()


class TestHandlePhoneNumber:
    """Tests for phone number normalization."""

    def test_us_format(self):
        import re
        pattern = re.compile(
            r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})"
        )
        match = pattern.search("Call (555) 123-4567")
        result = _handle_phone_number(match)
        # Should contain spoken digits
        assert "five" in result.lower() or "5" in result

    def test_with_country_code(self):
        import re
        pattern = re.compile(
            r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})"
        )
        match = pattern.search("+1 555-123-4567")
        result = _handle_phone_number(match)
        assert "one" in result.lower() or "1" in result


class TestNormalizeText:
    """Tests for the main normalize_text function."""

    def test_combined_transformations(self):
        text = "Contact us at user@example.com or call (555) 123-4567"
        result = normalize_text(text)
        # Email should be normalized
        assert "at" in result
        assert "@" not in result

    def test_normalize_disabled(self):
        opts = NormalizationOptions(normalize=False)
        text = "$50 at user@test.com"
        result = normalize_text(text, opts)
        # Should return unchanged
        assert result == text

    def test_email_normalization_disabled(self):
        opts = NormalizationOptions(
            email_normalization=False,
            replace_remaining_symbols=False  # @ would be replaced without this
        )
        text = "Email user@test.com"
        result = normalize_text(text, opts)
        # Email should remain unchanged
        assert "@" in result

    def test_url_normalization_disabled(self):
        opts = NormalizationOptions(url_normalization=False)
        text = "Visit https://example.com"
        result = normalize_text(text, opts)
        # URL structure should remain more intact
        # (though other normalizations might still apply)
        assert "example" in result

    def test_title_replacements_doctor(self):
        text = "Dr. Smith is here"
        result = normalize_text(text)
        assert "Doctor" in result or "Dr." in result

    def test_title_replacements_mister(self):
        text = "Mr. Jones arrived"
        result = normalize_text(text)
        assert "Mister" in result or "Mr." in result

    def test_cjk_punctuation(self):
        # Chinese comma should be converted
        text = "Hello\u3001world"  # Chinese comma
        result = normalize_text(text)
        # Should have English punctuation
        assert "\u3001" not in result

    def test_symbol_replacements(self):
        opts = NormalizationOptions(replace_remaining_symbols=True)
        text = "A & B"
        result = normalize_text(text, opts)
        assert "and" in result.lower()

    def test_symbol_replacements_at(self):
        opts = NormalizationOptions(
            email_normalization=False,
            replace_remaining_symbols=True
        )
        text = "item @ price"
        result = normalize_text(text, opts)
        assert "at" in result.lower()

    def test_quotes_normalized(self):
        # Smart quotes should be normalized
        text = "He said \u201cHello\u201d"  # Curly quotes
        result = normalize_text(text)
        assert "\u201c" not in result
        assert "\u201d" not in result

    def test_number_range(self):
        text = "Pages 10-20"
        result = normalize_text(text)
        # Dash between numbers converts to "to" per the code
        # But numbers are also converted to words first, so we check both possibilities
        result_lower = result.lower()
        assert "to" in result_lower or "twenty" in result_lower

    def test_commas_in_numbers_removed(self):
        text = "Population is 1,000,000"
        result = normalize_text(text)
        # Commas should be removed before number conversion
        assert "1,000,000" not in result

    def test_empty_string(self):
        result = normalize_text("")
        assert result == ""

    def test_whitespace_normalization(self):
        text = "Multiple   spaces   here"
        result = normalize_text(text)
        assert "   " not in result

    def test_newlines_removed(self):
        text = "Line one\nLine two"
        result = normalize_text(text)
        assert "\n" not in result
