"""
Basic usage example for streaming-tts package.

Run with:
    python examples/basic_usage.py
"""

from streaming_tts import TextToAudioStream, KokoroEngine


def main():
    # Initialize the TTS engine
    print("Initializing KokoroEngine...")
    engine = KokoroEngine(voice="af_heart")

    # Show available voices
    voices = engine.get_voices()
    print(f"Available voices: {len(voices)}")
    print(f"Using voice: {engine.current_voice}")

    # Create the audio stream
    stream = TextToAudioStream(engine)

    # Synthesize and play audio
    text = "Hello! This is a test of the streaming text to speech system."
    print(f"\nSpeaking: {text}\n")

    stream.feed(text).play()

    print("Done!")


if __name__ == "__main__":
    main()
