"""
Text processing utilities for TTS.
Includes pause tag support and smart text chunking.
"""

import re
from dataclasses import dataclass
from typing import Generator, List, Optional, Union

from .text_normalizer import NormalizationOptions, normalize_text


# Pause tag pattern: [pause:1.5s] or [pause:500ms]
PAUSE_PATTERN = re.compile(r"\[pause:(\d+(?:\.\d+)?)(s|ms)\]", re.IGNORECASE)

# Sentence splitting patterns
SENTENCE_END_PATTERN = re.compile(r"([.!?]+)\s+")
CHINESE_SENTENCE_PATTERN = re.compile(r"([\u3002\uff01\uff1f\uff0c\u3001]+)")


@dataclass
class TextChunk:
    """Represents a chunk of text or a pause."""
    text: str
    is_pause: bool = False
    pause_duration: float = 0.0  # Duration in seconds

    @property
    def is_text(self) -> bool:
        return not self.is_pause and bool(self.text.strip())


@dataclass
class ChunkingOptions:
    """Options for smart text chunking."""
    target_min_tokens: int = 175
    target_max_tokens: int = 250
    absolute_max_tokens: int = 450
    normalize: bool = True
    normalization_options: Optional[NormalizationOptions] = None


def parse_pause_tags(text: str) -> List[TextChunk]:
    """
    Parse text and extract pause tags.

    Args:
        text: Input text potentially containing [pause:Xs] tags

    Returns:
        List of TextChunk objects representing text and pauses

    Example:
        >>> parse_pause_tags("Hello. [pause:1.5s] World!")
        [TextChunk(text="Hello. ", is_pause=False),
         TextChunk(text="", is_pause=True, pause_duration=1.5),
         TextChunk(text=" World!", is_pause=False)]
    """
    chunks = []
    last_end = 0

    for match in PAUSE_PATTERN.finditer(text):
        # Add text before the pause
        if match.start() > last_end:
            text_before = text[last_end:match.start()]
            if text_before.strip():
                chunks.append(TextChunk(text=text_before))

        # Parse pause duration
        duration = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "ms":
            duration /= 1000  # Convert to seconds

        chunks.append(TextChunk(text="", is_pause=True, pause_duration=duration))
        last_end = match.end()

    # Add remaining text
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining.strip():
            chunks.append(TextChunk(text=remaining))

    # If no pauses found, return the original text as a single chunk
    if not chunks and text.strip():
        chunks.append(TextChunk(text=text))

    return chunks


def split_sentences(text: str, lang_code: str = "a") -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Input text
        lang_code: Language code ('a' for English, 'z' for Chinese, etc.)

    Returns:
        List of sentences
    """
    # Detect Chinese text
    is_chinese = lang_code.startswith("z") or bool(re.search(r"[\u4e00-\u9fff]", text))

    if is_chinese:
        parts = CHINESE_SENTENCE_PATTERN.split(text)
    else:
        parts = SENTENCE_END_PATTERN.split(text)

    sentences = []
    i = 0
    while i < len(parts):
        sentence = parts[i].strip()
        # Append punctuation if it exists
        if i + 1 < len(parts) and parts[i + 1]:
            sentence += parts[i + 1]
        if sentence:
            sentences.append(sentence.strip())
        i += 2 if not is_chinese else 2

    # Handle case where split didn't work well
    if not sentences and text.strip():
        sentences = [text.strip()]

    return sentences


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.

    This is a rough approximation. For English, ~4 characters per token.
    For better accuracy, use a proper tokenizer.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Rough estimation: ~4 chars per token for English
    return max(1, len(text) // 4)


def smart_split(
    text: str,
    lang_code: str = "a",
    options: Optional[ChunkingOptions] = None,
) -> Generator[TextChunk, None, None]:
    """
    Smart text splitting with pause tag support and token-aware chunking.

    This generator yields TextChunk objects that are either:
    - Text chunks optimized for TTS (targeting 175-250 tokens)
    - Pause chunks indicating silence duration

    Args:
        text: Input text (may contain [pause:Xs] tags)
        lang_code: Language code for sentence splitting
        options: Chunking options

    Yields:
        TextChunk objects

    Example:
        >>> for chunk in smart_split("Hello world. [pause:1s] How are you?"):
        ...     if chunk.is_pause:
        ...         # Insert silence
        ...         pass
        ...     else:
        ...         # Synthesize text
        ...         engine.synthesize(chunk.text)
    """
    if options is None:
        options = ChunkingOptions()

    # Step 1: Parse pause tags
    pause_chunks = parse_pause_tags(text)

    for chunk in pause_chunks:
        if chunk.is_pause:
            yield chunk
            continue

        # Step 2: Optionally normalize text
        processed_text = chunk.text
        if options.normalize:
            norm_opts = options.normalization_options or NormalizationOptions()
            processed_text = normalize_text(processed_text, norm_opts)

        # Step 3: Split into sentences
        sentences = split_sentences(processed_text, lang_code)

        # Step 4: Build optimal chunks
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = estimate_tokens(sentence)

            # Handle sentences that exceed max tokens
            if sentence_tokens > options.absolute_max_tokens:
                # Yield current chunk first
                if current_chunk:
                    yield TextChunk(text=" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence on commas
                clauses = re.split(r"([,;])", sentence)
                clause_chunk = []
                clause_tokens = 0

                for i in range(0, len(clauses), 2):
                    clause = clauses[i].strip()
                    punct = clauses[i + 1] if i + 1 < len(clauses) else ""

                    if not clause:
                        continue

                    full_clause = clause + punct
                    clause_token_count = estimate_tokens(full_clause)

                    if (clause_tokens + clause_token_count <= options.target_max_tokens
                        and clause_tokens + clause_token_count <= options.absolute_max_tokens):
                        clause_chunk.append(full_clause)
                        clause_tokens += clause_token_count
                    else:
                        if clause_chunk:
                            yield TextChunk(text=" ".join(clause_chunk))
                        clause_chunk = [full_clause]
                        clause_tokens = clause_token_count

                if clause_chunk:
                    yield TextChunk(text=" ".join(clause_chunk))
                continue

            # Regular sentence handling
            if (current_tokens >= options.target_min_tokens
                and current_tokens + sentence_tokens > options.target_max_tokens):
                # Yield current chunk and start new one
                yield TextChunk(text=" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            elif current_tokens + sentence_tokens <= options.target_max_tokens:
                # Keep building chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            elif (current_tokens + sentence_tokens <= options.absolute_max_tokens
                  and current_tokens < options.target_min_tokens):
                # Allow exceeding target if we haven't reached minimum
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Yield and start new
                if current_chunk:
                    yield TextChunk(text=" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens

        # Yield remaining chunk
        if current_chunk:
            yield TextChunk(text=" ".join(current_chunk))


def process_text_with_pauses(
    text: str,
    lang_code: str = "a",
    normalize: bool = True,
) -> Generator[Union[str, float], None, None]:
    """
    Simple generator that yields text strings and pause durations.

    This is a simpler interface than smart_split() for basic use cases.

    Args:
        text: Input text (may contain [pause:Xs] tags)
        lang_code: Language code
        normalize: Whether to normalize text

    Yields:
        Either a string (text to synthesize) or a float (pause duration in seconds)

    Example:
        >>> for item in process_text_with_pauses("Hello. [pause:1s] World!"):
        ...     if isinstance(item, float):
        ...         time.sleep(item)  # Pause
        ...     else:
        ...         engine.synthesize(item)  # Speak
    """
    options = ChunkingOptions(normalize=normalize)

    for chunk in smart_split(text, lang_code, options):
        if chunk.is_pause:
            yield chunk.pause_duration
        elif chunk.is_text:
            yield chunk.text
