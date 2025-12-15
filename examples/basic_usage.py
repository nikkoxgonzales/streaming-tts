"""Basic usage examples for streaming-tts.

Run with: python examples/basic_usage.py
Requires: pip install streaming-tts[playback]
"""

from streaming_tts import TTSStream, TTSConfig


def simple_playback():
    """Simplest possible usage - just speak text."""
    stream = TTSStream()
    stream.feed("Hello world! This is a simple text to speech example.").play()


def chained_playback():
    """Chain multiple text segments."""
    stream = TTSStream()
    stream.feed("First sentence.").feed("Second sentence.").feed("Third sentence.").play()


def custom_voice():
    """Use a different voice."""
    config = TTSConfig(voice="am_adam")  # Male American voice
    stream = TTSStream(config=config)
    stream.feed("This is Adam speaking.").play()


def custom_speed():
    """Adjust speech speed."""
    # Slower
    slow = TTSStream(config=TTSConfig(speed=0.8))
    slow.feed("This is slower speech.").play()

    # Faster
    fast = TTSStream(config=TTSConfig(speed=1.3))
    fast.feed("This is faster speech.").play()


def context_manager():
    """Use context manager for automatic cleanup."""
    with TTSStream() as stream:
        stream.feed("Using context manager for automatic resource cleanup.").play()
    # Resources automatically released here


def change_voice_mid_session():
    """Change voice between utterances."""
    stream = TTSStream()

    stream.feed("Hello, I'm the default voice.")
    stream.play()

    stream.set_voice("bf_alice")  # British female
    stream.feed("And now I'm Alice with a British accent.")
    stream.play()

    stream.set_voice("jf_alpha")  # Japanese female
    stream.feed("Konnichiwa!")
    stream.play()

    stream.shutdown()


def list_available_voices():
    """Print all available voices."""
    stream = TTSStream()
    voices = stream.get_voices()

    print("Available voices:")
    print("-" * 40)

    # Group by language
    lang_names = {
        "a": "American English",
        "b": "British English",
        "j": "Japanese",
        "z": "Mandarin Chinese",
        "e": "Spanish",
        "f": "French",
        "h": "Hindi",
        "i": "Italian",
        "p": "Brazilian Portuguese",
    }

    for prefix, lang in lang_names.items():
        lang_voices = [v for v in voices if v.startswith(prefix)]
        if lang_voices:
            print(f"\n{lang}:")
            for v in lang_voices:
                gender = "Female" if v[1] == "f" else "Male"
                name = v.split("_")[1] if "_" in v else v
                print(f"  {v} ({gender}: {name})")

    stream.shutdown()


if __name__ == "__main__":
    print("=== Simple Playback ===")
    simple_playback()

    print("\n=== Chained Playback ===")
    chained_playback()

    print("\n=== Custom Voice ===")
    custom_voice()

    print("\n=== Custom Speed ===")
    custom_speed()

    print("\n=== Context Manager ===")
    context_manager()

    print("\n=== Change Voice Mid-Session ===")
    change_voice_mid_session()

    print("\n=== Available Voices ===")
    list_available_voices()
