"""Configuration dataclasses for streaming-tts."""

from dataclasses import dataclass
from typing import Literal


# All available Kokoro voices
KOKORO_VOICES: tuple[str, ...] = (
    # American English (lang_code='a')
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    # British English (lang_code='b')
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    # Japanese (lang_code='j')
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    # Mandarin Chinese (lang_code='z')
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    # Spanish (lang_code='e')
    "ef_dora", "em_alex", "em_santa",
    # French (lang_code='f')
    "ff_siwis",
    # Hindi (lang_code='h')
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    # Italian (lang_code='i')
    "if_sara", "im_nicola",
    # Brazilian Portuguese (lang_code='p')
    "pf_dora", "pm_alex", "pm_santa",
)

LangCode = Literal["a", "b", "j", "z", "e", "f", "h", "i", "p"]


@dataclass(frozen=True, slots=True)
class TTSConfig:
    """
    Immutable TTS configuration.

    Attributes:
        voice: Voice name or blend formula. Supported formats:
            - Single voice: "af_heart"
            - Old blend: "0.3*af_sarah + 0.7*am_adam"
            - New blend: "af_sarah+af_jessica" (equal weights)
            - New weighted: "af_sarah(0.3)+af_jessica(0.7)"
            - Subtraction: "af_sarah-af_jessica"
        speed: Speech speed multiplier (1.0 = normal)
        sample_rate: Audio sample rate in Hz (Kokoro uses 24000)
        channels: Number of audio channels (1 = mono)
        device: Device to run on ("cuda", "mps", "cpu", or None for auto-detect)
        trim_silence: Whether to trim leading/trailing silence
        silence_threshold: Amplitude threshold for silence detection
        fade_in_ms: Fade-in duration in milliseconds after trim
        fade_out_ms: Fade-out duration in milliseconds after trim
        extra_trim_start_ms: Extra milliseconds to trim from start
        extra_trim_end_ms: Extra milliseconds to trim from end
        memory_threshold_gb: GPU memory threshold for auto-clearing (CUDA only)
    """

    voice: str = "af_heart"
    speed: float = 1.0
    sample_rate: int = 24000
    channels: int = 1
    device: str | None = None
    trim_silence: bool = True
    silence_threshold: float = 0.005
    fade_in_ms: int = 10
    fade_out_ms: int = 10
    extra_trim_start_ms: int = 15
    extra_trim_end_ms: int = 15
    memory_threshold_gb: float = 2.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.speed <= 0:
            raise ValueError(f"speed must be positive, got {self.speed}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {self.channels}")
        if self.silence_threshold < 0:
            raise ValueError(f"silence_threshold must be non-negative, got {self.silence_threshold}")
        if self.device is not None and self.device not in ("cuda", "mps", "cpu"):
            raise ValueError(f"device must be 'cuda', 'mps', 'cpu', or None, got {self.device}")


@dataclass(frozen=True, slots=True)
class PlaybackConfig:
    """
    Audio playback configuration.

    Attributes:
        device_index: PyAudio device index (None = default device)
        frames_per_buffer: Buffer size for PyAudio stream
        muted: If True, skip actual audio output (still generates chunks)
    """

    device_index: int | None = None
    frames_per_buffer: int = 512
    muted: bool = False


def get_lang_code(voice: str) -> LangCode:
    """
    Determine language code from voice name.

    Handles single voices and all blend formula formats:
    - Old: "0.3*af_sarah + 0.7*am_adam"
    - New: "af_sarah+af_jessica", "af_sarah(0.3)+af_jessica(0.7)"
    - Subtraction: "af_sarah-af_jessica"

    Args:
        voice: Voice name or blend formula

    Returns:
        Single-character language code
    """
    import re

    # Handle old-style formulas: "0.3*af_sarah + 0.7*am_adam"
    if "*" in voice:
        parts = voice.replace("+", " ").replace("-", " ").split()
        for part in parts:
            if "*" in part:
                voice_token = part.split("*")[-1].strip()
                return get_lang_code(voice_token)
        return "a"  # fallback

    # Handle new-style formulas: "af_sarah+af_jessica" or "af_sarah(0.3)+af_jessica(0.7)"
    if "+" in voice or "-" in voice:
        # Split on + or - but keep them
        parts = re.split(r"[+-]", voice)
        if parts:
            # Get first voice name (strip weight if present)
            first_voice = parts[0].strip()
            if "(" in first_voice:
                first_voice = first_voice.split("(")[0].strip()
            return get_lang_code(first_voice)
        return "a"

    # Strip weight suffix if present: "af_sarah(0.3)" -> "af_sarah"
    if "(" in voice:
        voice = voice.split("(")[0].strip()

    # Standard single-voice mapping
    prefix = voice[:2].lower() if len(voice) >= 2 else ""

    prefix_map: dict[str, LangCode] = {
        "af": "a", "am": "a",  # American English
        "bf": "b", "bm": "b",  # British English
        "jf": "j", "jm": "j",  # Japanese
        "zf": "z", "zm": "z",  # Mandarin Chinese
        "ef": "e", "em": "e",  # Spanish
        "ff": "f",             # French
        "hf": "h", "hm": "h",  # Hindi
        "if": "i", "im": "i",  # Italian
        "pf": "p", "pm": "p",  # Brazilian Portuguese
    }

    if prefix in prefix_map:
        return prefix_map[prefix]

    # Fallback to first character if valid
    first_char = voice[0].lower() if voice else "a"
    if first_char in "abjzefhip":
        return first_char  # type: ignore[return-value]
    return "a"
