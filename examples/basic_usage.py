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
    text = (
        "This is a warming up of the streaming text to speech engine."
        "Hey, how's it going? I wanted to tell you something important: we finally did it!"
        "So, the project took three months; longer than expected, honestly. But here's the thing... it was worth every single hour."
        "You know what surprised me? The team. They worked hard, stayed focused, and never complained once."
        "Now, let me ask you this: are you ready for the next challenge? Because trust me, it's going to be bigger, faster, and way more exciting!"
        "Wait... did you hear that? Never mind, it was nothing."
        "Anyway, here's the plan: first, we regroup; second, we strategize; third, we execute. Simple, right?"
        "Oh! One more thing. Don't forget to celebrate the small wins; they matter more than you think."
        "Alright, that's it for now. Talk soon... take care!"
    )
    print(f"\nSpeaking: {text}\n")

    stream.feed(text).play()

    print("Done!")


if __name__ == "__main__":
    main()
