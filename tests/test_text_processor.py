"""Tests for streaming_tts.text_processor module."""

from streaming_tts.text_processor import (
    parse_pause_tags,
    split_sentences,
    estimate_tokens,
    smart_split,
    process_text_with_pauses,
    TextChunk,
    ChunkingOptions,
)


class TestTextChunk:
    """Tests for TextChunk dataclass."""

    def test_text_chunk_defaults(self):
        chunk = TextChunk(text="Hello")
        assert chunk.text == "Hello"
        assert chunk.is_pause is False
        assert chunk.pause_duration == 0.0

    def test_text_chunk_is_text_property(self):
        chunk = TextChunk(text="Hello")
        assert chunk.is_text is True

    def test_text_chunk_whitespace_only_not_text(self):
        chunk = TextChunk(text="   ")
        assert chunk.is_text is False

    def test_text_chunk_empty_not_text(self):
        chunk = TextChunk(text="")
        assert chunk.is_text is False

    def test_pause_chunk(self):
        chunk = TextChunk(text="", is_pause=True, pause_duration=1.5)
        assert chunk.is_pause is True
        assert chunk.pause_duration == 1.5
        assert chunk.is_text is False


class TestChunkingOptions:
    """Tests for ChunkingOptions dataclass."""

    def test_default_values(self):
        opts = ChunkingOptions()
        assert opts.target_min_tokens == 175
        assert opts.target_max_tokens == 250
        assert opts.absolute_max_tokens == 450
        assert opts.normalize is True
        assert opts.normalization_options is None

    def test_custom_values(self):
        opts = ChunkingOptions(
            target_min_tokens=100,
            target_max_tokens=200,
            normalize=False
        )
        assert opts.target_min_tokens == 100
        assert opts.target_max_tokens == 200
        assert opts.normalize is False


class TestParsePauseTags:
    """Tests for parse_pause_tags function."""

    def test_no_pause_tags(self):
        result = parse_pause_tags("Hello world")
        assert len(result) == 1
        assert result[0].text == "Hello world"
        assert result[0].is_pause is False

    def test_single_pause_seconds(self):
        result = parse_pause_tags("Hello. [pause:1.5s] World!")
        assert len(result) == 3
        assert result[0].text == "Hello. "
        assert result[0].is_pause is False
        assert result[1].is_pause is True
        assert result[1].pause_duration == 1.5
        assert result[2].text == " World!"

    def test_single_pause_milliseconds(self):
        result = parse_pause_tags("Hello [pause:500ms] there")
        assert len(result) == 3
        assert result[1].is_pause is True
        assert result[1].pause_duration == 0.5

    def test_multiple_pauses(self):
        result = parse_pause_tags("A [pause:1s] B [pause:2s] C")
        assert len(result) == 5
        assert result[0].text == "A "
        assert result[1].is_pause is True
        assert result[1].pause_duration == 1.0
        assert result[2].text == " B "
        assert result[3].is_pause is True
        assert result[3].pause_duration == 2.0
        assert result[4].text == " C"

    def test_pause_at_start(self):
        result = parse_pause_tags("[pause:1s] Hello")
        assert len(result) == 2
        assert result[0].is_pause is True
        assert result[0].pause_duration == 1.0
        assert result[1].text == " Hello"

    def test_pause_at_end(self):
        result = parse_pause_tags("Hello [pause:1s]")
        assert len(result) == 2
        assert result[0].text == "Hello "
        assert result[1].is_pause is True

    def test_case_insensitive_uppercase(self):
        result = parse_pause_tags("[PAUSE:1S] Hello")
        assert len(result) == 2
        assert result[0].is_pause is True
        assert result[0].pause_duration == 1.0

    def test_case_insensitive_mixed(self):
        result = parse_pause_tags("[Pause:500Ms] Hello")
        assert len(result) == 2
        assert result[0].is_pause is True
        assert result[0].pause_duration == 0.5

    def test_decimal_seconds(self):
        result = parse_pause_tags("[pause:0.25s] text")
        assert result[0].is_pause is True
        assert result[0].pause_duration == 0.25

    def test_empty_string(self):
        result = parse_pause_tags("")
        assert len(result) == 0

    def test_whitespace_only(self):
        result = parse_pause_tags("   ")
        assert len(result) == 0


class TestSplitSentences:
    """Tests for split_sentences function."""

    def test_simple_english(self):
        result = split_sentences("Hello. World. Test.")
        assert len(result) == 3
        assert "Hello." in result[0]
        assert "World." in result[1]
        assert "Test." in result[2]

    def test_multiple_punctuation(self):
        result = split_sentences("Really?! Yes!!")
        assert len(result) == 2

    def test_no_punctuation(self):
        result = split_sentences("Hello world no punctuation")
        assert len(result) == 1
        assert result[0] == "Hello world no punctuation"

    def test_empty_string(self):
        result = split_sentences("")
        assert len(result) == 0

    def test_single_sentence(self):
        result = split_sentences("Just one sentence here.")
        assert len(result) == 1

    def test_question_marks(self):
        result = split_sentences("What is this? Another question?")
        assert len(result) >= 1

    def test_chinese_text(self):
        # Chinese text with Chinese period
        result = split_sentences("你好。世界。", lang_code="z")
        assert len(result) >= 1

    def test_lang_code_chinese_detected(self):
        # Auto-detect Chinese characters
        result = split_sentences("这是测试。下一句。")
        assert len(result) >= 1


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_short_text(self):
        result = estimate_tokens("Hi")
        assert result >= 1  # Minimum is 1

    def test_typical_sentence(self):
        # "This is a test sentence" = ~23 chars / 4 = ~5 tokens
        result = estimate_tokens("This is a test sentence")
        assert 4 <= result <= 8

    def test_longer_text(self):
        text = "This is a much longer sentence with many more words and characters."
        result = estimate_tokens(text)
        # ~68 chars / 4 = ~17 tokens
        assert 10 <= result <= 25

    def test_whitespace_normalization(self):
        result1 = estimate_tokens("hello world")
        result2 = estimate_tokens("hello   world")
        assert result1 == result2

    def test_empty_string(self):
        result = estimate_tokens("")
        assert result == 1  # max(1, 0) = 1


class TestSmartSplit:
    """Tests for smart_split generator function."""

    def test_short_text_single_chunk(self):
        chunks = list(smart_split("Hello world."))
        assert len(chunks) >= 1
        assert all(isinstance(c, TextChunk) for c in chunks)
        # Short text should yield text chunks
        text_chunks = [c for c in chunks if c.is_text]
        assert len(text_chunks) >= 1

    def test_handles_pauses(self):
        chunks = list(smart_split("Hello. [pause:1s] World."))
        pause_chunks = [c for c in chunks if c.is_pause]
        text_chunks = [c for c in chunks if c.is_text]
        assert len(pause_chunks) == 1
        assert pause_chunks[0].pause_duration == 1.0
        assert len(text_chunks) >= 1

    def test_respects_chunking_options(self):
        opts = ChunkingOptions(
            target_min_tokens=10,
            target_max_tokens=20,
            normalize=False
        )
        # Create text that would need splitting with these options
        text = "First sentence here. Second sentence there. Third one too. And fourth."
        chunks = list(smart_split(text, options=opts))
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_empty_text(self):
        chunks = list(smart_split(""))
        # Empty text should produce no chunks
        assert len(chunks) == 0

    def test_normalization_applied(self):
        # With normalization, numbers should be converted
        opts = ChunkingOptions(normalize=True)
        chunks = list(smart_split("I have $50.", options=opts))
        combined = " ".join(c.text for c in chunks if c.is_text)
        # After normalization, "$50" should be converted to words
        assert "50" not in combined or "dollar" in combined.lower() or "fifty" in combined.lower()

    def test_normalization_disabled(self):
        opts = ChunkingOptions(normalize=False)
        chunks = list(smart_split("I have $50.", options=opts))
        combined = " ".join(c.text for c in chunks if c.is_text)
        assert "$50" in combined


class TestProcessTextWithPauses:
    """Tests for process_text_with_pauses generator function."""

    def test_yields_strings_and_floats(self):
        results = list(process_text_with_pauses("Hello [pause:1s] World"))
        # Should yield text, pause duration, text
        has_str = any(isinstance(r, str) for r in results)
        has_float = any(isinstance(r, float) for r in results)
        assert has_str
        assert has_float

    def test_pause_duration_value(self):
        results = list(process_text_with_pauses("Hello [pause:2s] World"))
        floats = [r for r in results if isinstance(r, float)]
        assert 2.0 in floats

    def test_no_pauses(self):
        results = list(process_text_with_pauses("Hello world no pauses"))
        # Should only yield strings
        assert all(isinstance(r, str) for r in results)

    def test_normalize_parameter(self):
        results_normalized = list(process_text_with_pauses("I have $10.", normalize=True))
        results_raw = list(process_text_with_pauses("I have $10.", normalize=False))
        # Results should differ when normalization is toggled
        str_normalized = " ".join(r for r in results_normalized if isinstance(r, str))
        str_raw = " ".join(r for r in results_raw if isinstance(r, str))
        # Raw should contain $10, normalized should not
        assert "$10" in str_raw
        # Normalized should have "ten" or "dollar" instead
        assert "$10" not in str_normalized or "ten" in str_normalized.lower()

    def test_empty_string(self):
        results = list(process_text_with_pauses(""))
        assert len(results) == 0
