"""
Advanced usage example for streaming-tts package.
Demonstrates new features: pause tags, text normalization, smart chunking, and multi-format output.

Run with:
    python examples/advanced_usage.py
"""

import time
from streaming_tts import (
    TextToAudioStream,
    KokoroEngine,
    # Text processing
    normalize_text,
    NormalizationOptions,
    smart_split,
    parse_pause_tags,
    process_text_with_pauses,
    # Audio utilities
    AudioNormalizer,
    create_silence,
)


def demo_pause_tags():
    """Demonstrate pause tag support."""
    print("\n" + "=" * 60)
    print("DEMO: Pause Tags")
    print("=" * 60)

    text = "Hello there! [pause:1s] How are you doing today? [pause:0.5s] I hope you're well."
    print(f"Input: {text}\n")

    # Parse pause tags
    chunks = parse_pause_tags(text)
    for chunk in chunks:
        if chunk.is_pause:
            print(f"  [PAUSE: {chunk.pause_duration}s]")
        else:
            print(f"  Text: {chunk.text!r}")


def demo_text_normalization():
    """Demonstrate text normalization."""
    print("\n" + "=" * 60)
    print("DEMO: Text Normalization")
    print("=" * 60)

    examples = [
        "Contact me at user@example.com",
        "Visit https://github.com/project for more info",
        "The price is $42.50",
        "It happened in 1984",
        "The file is 2.5gb in size",
        "Call me at (555) 123-4567",
        "The meeting is at 3:30 PM",
    ]

    options = NormalizationOptions(normalize=True)

    for text in examples:
        normalized = normalize_text(text, options)
        print(f"  {text}")
        print(f"  -> {normalized}\n")


def demo_smart_chunking():
    """Demonstrate smart text chunking."""
    print("\n" + "=" * 60)
    print("DEMO: Smart Chunking")
    print("=" * 60)

    long_text = """
    Artificial intelligence has transformed many aspects of our daily lives.
    From virtual assistants to recommendation systems, AI is everywhere.
    [pause:0.5s]
    Machine learning, a subset of AI, enables computers to learn from data.
    Deep learning takes this further with neural networks that mimic the human brain.
    [pause:1s]
    The future of AI is both exciting and challenging. We must consider ethics,
    privacy, and the impact on employment as these technologies advance.
    """

    print("Input text (with pause tags):\n")
    print(long_text.strip())
    print("\nSmart-split chunks:\n")

    for i, chunk in enumerate(smart_split(long_text, lang_code="a"), 1):
        if chunk.is_pause:
            print(f"  Chunk {i}: [PAUSE {chunk.pause_duration}s]")
        else:
            tokens = len(chunk.text) // 4  # Rough estimate
            print(f"  Chunk {i} (~{tokens} tokens): {chunk.text[:80]}...")


def demo_audio_normalizer():
    """Demonstrate audio normalizer."""
    print("\n" + "=" * 60)
    print("DEMO: Audio Normalizer")
    print("=" * 60)

    normalizer = AudioNormalizer(
        sample_rate=24000,
        gap_trim_ms=1.0,
        padding_ms=410.0,
        silence_threshold_db=-45.0,
    )

    print("AudioNormalizer settings:")
    print(f"  Sample rate: {normalizer.sample_rate} Hz")
    print(f"  Gap trim: {normalizer.gap_trim_ms} ms")
    print(f"  Padding: {normalizer.padding_ms} ms")
    print(f"  Silence threshold: {normalizer.silence_threshold_db} dB")
    print("\nPunctuation padding multipliers:")
    for punct, mult in AudioNormalizer.PUNCTUATION_PADDING.items():
        print(f"  '{punct}' -> {mult}x padding")


def demo_full_synthesis():
    """Full synthesis demo with all features."""
    print("\n" + "=" * 60)
    print("DEMO: Full Synthesis with All Features")
    print("=" * 60)

    # Initialize engine
    print("Initializing KokoroEngine...")
    engine = KokoroEngine(voice="af_heart")

    # Create stream
    stream = TextToAudioStream(engine)

    # Text with pause tags and content that will be normalized
    text = """
    Welcome to the streaming TTS demo! [pause:0.5s]
    Today's date is January 15th, 2025.
    For more information, visit https://example.com.
    [pause:1s]
    The total cost is $99.99. Thank you for listening!
    """

    print(f"\nSynthesizing text with pause tags and normalization...")
    print(f"Input:\n{text.strip()}\n")

    # Process with pause support
    for item in process_text_with_pauses(text, normalize=True):
        if isinstance(item, float):
            print(f"  [Pausing for {item}s...]")
            time.sleep(item)
        else:
            print(f"  Speaking: {item[:60]}...")
            stream.feed(item).play()

    print("\nDone!")


def main():
    """Run all demos."""
    print("=" * 60)
    print("streaming-tts v0.2.0 - Advanced Features Demo")
    print("=" * 60)

    # Run demos that don't require audio playback
    demo_pause_tags()
    demo_text_normalization()
    demo_smart_chunking()
    demo_audio_normalizer()

    # Ask before running synthesis demo
    print("\n" + "=" * 60)
    response = input("Run full synthesis demo with audio playback? [y/N]: ")
    if response.lower() == 'y':
        demo_full_synthesis()
    else:
        print("Skipped synthesis demo.")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
