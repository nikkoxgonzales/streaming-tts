"""Voice blending examples for streaming-tts.

Demonstrates the various voice combination syntaxes supported
by the library for creating unique blended voices.

Run with: python examples/voice_blending.py
Requires: pip install streaming-tts[playback]
"""

from streaming_tts import TTSStream, TTSConfig


def equal_blend_new_syntax():
    """Blend two voices equally using new syntax (recommended)."""
    print("Voice: af_sarah+af_jessica (50/50 blend)")

    config = TTSConfig(voice="af_sarah+af_jessica")
    stream = TTSStream(config=config)
    stream.feed("This voice is an equal blend of Sarah and Jessica.").play()
    stream.shutdown()


def weighted_blend_new_syntax():
    """Blend voices with explicit weights using new syntax."""
    print("Voice: af_sarah(0.3)+af_jessica(0.7) (30/70 blend)")

    config = TTSConfig(voice="af_sarah(0.3)+af_jessica(0.7)")
    stream = TTSStream(config=config)
    stream.feed("This voice is thirty percent Sarah and seventy percent Jessica.").play()
    stream.shutdown()


def weighted_blend_old_syntax():
    """Blend voices using the classic weighted syntax."""
    print("Voice: 0.3*af_sarah + 0.7*am_adam (old syntax)")

    config = TTSConfig(voice="0.3*af_sarah + 0.7*am_adam")
    stream = TTSStream(config=config)
    stream.feed("This blends a female and male voice using the old syntax.").play()
    stream.shutdown()


def three_voice_blend():
    """Blend three or more voices together."""
    print("Voice: af_heart+af_nova+af_sky (three-way blend)")

    config = TTSConfig(voice="af_heart+af_nova+af_sky")
    stream = TTSStream(config=config)
    stream.feed("This voice combines three different female voices.").play()
    stream.shutdown()


def subtraction_blend():
    """Experimental voice subtraction."""
    print("Voice: af_sarah-af_jessica (subtraction - experimental)")

    config = TTSConfig(voice="af_sarah-af_jessica")
    stream = TTSStream(config=config)
    stream.feed("Voice subtraction creates unique characteristics.").play()
    stream.shutdown()


def dynamic_voice_blending():
    """Change voice blends dynamically during a session."""
    stream = TTSStream()

    blends = [
        ("af_heart", "Pure af_heart voice"),
        ("af_heart+am_adam", "Blended with am_adam"),
        ("af_heart(0.8)+am_adam(0.2)", "Mostly af_heart with a hint of am_adam"),
        ("am_adam", "Pure am_adam voice"),
    ]

    for voice, description in blends:
        print(f"\nVoice: {voice}")
        stream.set_voice(voice)
        stream.feed(description).play()

    stream.shutdown()


def compare_blend_ratios():
    """Compare different blend ratios of the same two voices."""
    stream = TTSStream()

    text = "Hello, this is a voice blending test."

    ratios = [
        ("af_sarah", "100% Sarah, 0% Adam"),
        ("af_sarah(0.75)+am_adam(0.25)", "75% Sarah, 25% Adam"),
        ("af_sarah(0.5)+am_adam(0.5)", "50% Sarah, 50% Adam"),
        ("af_sarah(0.25)+am_adam(0.75)", "25% Sarah, 75% Adam"),
        ("am_adam", "0% Sarah, 100% Adam"),
    ]

    for voice, description in ratios:
        print(f"\n{description}")
        stream.set_voice(voice)
        stream.feed(text).play()

    stream.shutdown()


def cross_language_blend():
    """Blend voices from different languages (experimental)."""
    print("\nCross-language blending (experimental):")

    # Note: Cross-language blending may produce unexpected results
    # as the voice characteristics are trained for specific languages

    stream = TTSStream()

    # American + British blend
    stream.set_voice("af_sarah+bf_alice")
    print("Voice: af_sarah+bf_alice (American + British)")
    stream.feed("A transatlantic accent blend.").play()

    stream.shutdown()


def save_blended_voice_audio():
    """Save blended voice output to file for comparison."""
    import wave
    from pathlib import Path

    voices = [
        ("af_heart", "heart_pure.wav"),
        ("af_heart+am_adam", "heart_adam_blend.wav"),
    ]

    text = "Testing voice blend audio output."
    config_base = TTSConfig(voice="af_heart")

    for voice, filename in voices:
        chunks = []

        def collect(chunk: bytes):
            chunks.append(chunk)

        config = TTSConfig(voice=voice)
        stream = TTSStream(config=config)
        stream.feed(text)
        stream.play(on_chunk=collect, muted=True)

        # Save as WAV
        audio_data = b"".join(chunks)
        output = Path(filename)
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(audio_data)

        print(f"Saved {voice} -> {output} ({len(audio_data)} bytes)")
        stream.shutdown()


def list_blendable_voices():
    """Show voices that work well for blending."""
    print("\nVoices by language (can blend within same language):")
    print("-" * 50)

    voices = TTSStream().get_voices()

    languages = {
        "American English (a)": [v for v in voices if v.startswith(("af_", "am_"))],
        "British English (b)": [v for v in voices if v.startswith(("bf_", "bm_"))],
        "Japanese (j)": [v for v in voices if v.startswith(("jf_", "jm_"))],
        "Chinese (z)": [v for v in voices if v.startswith(("zf_", "zm_"))],
        "Spanish (e)": [v for v in voices if v.startswith(("ef_", "em_"))],
        "French (f)": [v for v in voices if v.startswith("ff_")],
        "Hindi (h)": [v for v in voices if v.startswith(("hf_", "hm_"))],
        "Italian (i)": [v for v in voices if v.startswith(("if_", "im_"))],
        "Portuguese (p)": [v for v in voices if v.startswith(("pf_", "pm_"))],
    }

    for lang, lang_voices in languages.items():
        if lang_voices:
            print(f"\n{lang}:")
            female = [v for v in lang_voices if v[1] == "f"]
            male = [v for v in lang_voices if v[1] == "m"]
            if female:
                print(f"  Female: {', '.join(female)}")
            if male:
                print(f"  Male:   {', '.join(male)}")


def main():
    print("=== Equal Blend (New Syntax) ===")
    equal_blend_new_syntax()

    print("\n=== Weighted Blend (New Syntax) ===")
    weighted_blend_new_syntax()

    print("\n=== Weighted Blend (Old Syntax) ===")
    weighted_blend_old_syntax()

    print("\n=== Three Voice Blend ===")
    three_voice_blend()

    print("\n=== Subtraction Blend (Experimental) ===")
    subtraction_blend()

    print("\n=== Dynamic Voice Blending ===")
    dynamic_voice_blending()

    print("\n=== Compare Blend Ratios ===")
    compare_blend_ratios()

    print("\n=== Cross-Language Blend ===")
    cross_language_blend()

    print("\n=== Save Blended Voice Audio ===")
    save_blended_voice_audio()

    print("\n=== Blendable Voices Reference ===")
    list_blendable_voices()


if __name__ == "__main__":
    main()
