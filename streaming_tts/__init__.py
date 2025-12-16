"""streaming-tts: Lightweight streaming text-to-speech with Kokoro engine."""

import warnings

# Suppress PyTorch warnings from Kokoro model internals
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last")
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

from .text_to_stream import TextToAudioStream
from .kokoro_engine import KokoroEngine, KokoroVoice
from .base_engine import BaseEngine, TimingInfo

# Text processing
from .text_normalizer import normalize_text, NormalizationOptions
from .text_processor import (
    smart_split,
    parse_pause_tags,
    process_text_with_pauses,
    TextChunk,
    ChunkingOptions,
)

# Audio utilities
from .audio_writer import (
    StreamingAudioWriter,
    AudioNormalizer,
    create_silence,
)

__version__ = "0.2.0"

__all__ = [
    # Version
    "__version__",
    # Core
    "TextToAudioStream",
    "KokoroEngine",
    "KokoroVoice",
    "BaseEngine",
    "TimingInfo",
    # Text processing
    "normalize_text",
    "NormalizationOptions",
    "smart_split",
    "parse_pause_tags",
    "process_text_with_pauses",
    "TextChunk",
    "ChunkingOptions",
    # Audio utilities
    "StreamingAudioWriter",
    "AudioNormalizer",
    "create_silence",
]
