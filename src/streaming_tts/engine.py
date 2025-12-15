"""Kokoro TTS engine wrapper with GPU support and enhanced voice blending."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .config import TTSConfig, get_lang_code

if TYPE_CHECKING:
    from kokoro import KModel, KPipeline


@dataclass(frozen=True, slots=True)
class TimingInfo:
    """Word-level timing information from TTS synthesis."""

    start_time: float
    end_time: float
    word: str


class KokoroTTS:
    """
    Kokoro TTS engine with GPU support and enhanced voice blending.

    Features:
    - Auto-detects best device (CUDA > MPS > CPU)
    - GPU memory management with OOM retry
    - Enhanced voice blending syntax
    - Efficient model sharing across language pipelines

    Example:
        engine = KokoroTTS(TTSConfig(voice="af_heart"))
        engine.load()
        for chunk in engine.synthesize("Hello world"):
            process(chunk)
        engine.shutdown()

    Voice blending formats:
        - Single: "af_heart"
        - Old style: "0.3*af_sarah + 0.7*am_adam"
        - New style: "af_sarah+af_jessica" (equal weights)
        - Weighted: "af_sarah(0.3)+af_jessica(0.7)"
        - Subtraction: "af_sarah-af_jessica"
    """

    def __init__(self, config: TTSConfig | None = None) -> None:
        """
        Initialize the Kokoro TTS engine.

        Args:
            config: TTS configuration. Uses defaults if not provided.
        """
        self.config = config or TTSConfig()
        self._model: KModel | None = None
        self._pipelines: dict[str, KPipeline] = {}
        self._blended_voices: dict[str, torch.Tensor] = {}
        self._current_voice = self.config.voice
        self._current_lang = get_lang_code(self.config.voice)
        self._loaded = False
        self._device = self._detect_device()

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        if self.config.device is not None:
            return self.config.device

        if torch.cuda.is_available():  # pragma: no cover
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover
            return "mps"
        return "cpu"

    def load(self) -> None:
        """
        Lazy-load the Kokoro model and initial pipeline.

        Downloads model weights on first call if not cached.
        """
        if self._loaded:
            return

        from kokoro import KPipeline

        # Create pipeline - it will handle model loading internally
        # kokoro 0.9.x uses repo_id for auto-download
        self._pipelines[self._current_lang] = KPipeline(
            lang_code=self._current_lang,
            repo_id="hexgrad/Kokoro-82M",
        )
        self._loaded = True

    def _get_pipeline(self, lang_code: str) -> KPipeline:
        """Get or create pipeline for language code."""
        if lang_code not in self._pipelines:
            from kokoro import KPipeline

            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id="hexgrad/Kokoro-82M",
            )
        return self._pipelines[lang_code]

    def _check_memory(self) -> bool:
        """Check if GPU memory usage exceeds threshold."""
        if self._device == "cuda":  # pragma: no cover
            memory_gb = torch.cuda.memory_allocated() / 1e9
            return memory_gb > self.config.memory_threshold_gb
        return False

    def _clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if self._device == "cuda":  # pragma: no cover
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self._device == "mps" and hasattr(torch.mps, "empty_cache"):  # pragma: no cover
            torch.mps.empty_cache()

    def _parse_voice_formula(
        self, formula: str, pipeline: KPipeline
    ) -> torch.Tensor:
        """
        Parse a voice formula and return the blended voice tensor.

        Supports multiple formats:
        - Old: "0.3*af_sarah + 0.7*am_adam"
        - New: "af_sarah+af_jessica" (equal blend)
        - Weighted: "af_sarah(0.3)+af_jessica(0.7)"
        - Subtraction: "af_sarah-af_jessica"

        Args:
            formula: Voice blend formula
            pipeline: Kokoro pipeline for loading voices

        Returns:
            Blended voice tensor
        """
        if formula in self._blended_voices:
            return self._blended_voices[formula]

        # Detect formula type and parse
        if "*" in formula:
            # Old style: "0.3*af_sarah + 0.7*am_adam"
            result = self._parse_old_style_formula(formula, pipeline)
        else:
            # New style: "af+am", "af(0.3)+am(0.7)", "af-am"
            result = self._parse_new_style_formula(formula, pipeline)

        self._blended_voices[formula] = result
        return result

    def _parse_old_style_formula(
        self, formula: str, pipeline: KPipeline
    ) -> torch.Tensor:
        """Parse old-style formula: 0.3*af_sarah + 0.7*am_adam"""
        # Split on + and - while tracking operation
        parts = re.split(r"\s*([+-])\s*", formula)

        sum_tensor: torch.Tensor | None = None
        total_weight = 0.0
        current_op = "+"

        for part in parts:
            part = part.strip()
            if part in ("+", "-"):
                current_op = part
                continue
            if not part:
                continue

            if "*" not in part:
                raise ValueError(f"Malformed formula segment (missing '*'): '{part}'")

            weight_str, voice_name = part.split("*", 1)
            weight = float(weight_str.strip())
            voice_name = voice_name.strip()

            voice_tensor = pipeline.load_single_voice(voice_name)

            if current_op == "-":
                weight = -weight

            total_weight += abs(weight)
            weighted = weight * voice_tensor

            sum_tensor = weighted if sum_tensor is None else sum_tensor + weighted

        if total_weight == 0 or sum_tensor is None:
            raise ValueError(f"Invalid voice formula: {formula}")

        return sum_tensor / total_weight

    def _parse_new_style_formula(
        self, formula: str, pipeline: KPipeline
    ) -> torch.Tensor:
        """
        Parse new-style formula.

        Formats:
        - "af_sarah+af_jessica" (equal weights)
        - "af_sarah(0.3)+af_jessica(0.7)" (explicit weights)
        - "af_sarah-af_jessica" (subtraction)
        """
        # Split on + and - while tracking operation
        parts = re.split(r"([+-])", formula)

        voices: list[tuple[str, float, str]] = []  # (voice_name, weight, operation)
        current_op = "+"

        for part in parts:
            part = part.strip()
            if part in ("+", "-"):
                current_op = part
                continue
            if not part:
                continue

            # Parse voice name and optional weight
            if "(" in part and ")" in part:
                # "af_sarah(0.3)"
                match = re.match(r"([^(]+)\(([^)]+)\)", part)
                if match:
                    voice_name = match.group(1).strip()
                    weight = float(match.group(2))
                else:
                    raise ValueError(f"Malformed voice weight: '{part}'")
            else:
                voice_name = part
                weight = 1.0

            voices.append((voice_name, weight, current_op))

        if not voices:
            raise ValueError(f"No voices found in formula: {formula}")

        # Normalize weights if no explicit weights given
        has_explicit_weights = any("(" in formula for _ in [1])
        total_weight = sum(v[1] for v in voices)

        sum_tensor: torch.Tensor | None = None

        for voice_name, weight, op in voices:
            voice_tensor = pipeline.load_single_voice(voice_name)

            # Normalize weight
            if has_explicit_weights:
                normalized_weight = weight / total_weight if total_weight > 0 else weight
            else:
                normalized_weight = 1.0 / len(voices)

            if op == "-":
                normalized_weight = -normalized_weight

            weighted = normalized_weight * voice_tensor

            sum_tensor = weighted if sum_tensor is None else sum_tensor + weighted

        if sum_tensor is None:
            raise ValueError(f"Failed to blend voices: {formula}")

        return sum_tensor

    def _is_blend_formula(self, voice: str) -> bool:
        """Check if voice string is a blend formula."""
        return "*" in voice or "+" in voice or "-" in voice

    def synthesize(
        self,
        text: str,
        *,
        yield_timing: bool = False,
    ) -> Iterator[np.ndarray | tuple[np.ndarray, list[TimingInfo]]]:
        """
        Synthesize text to audio chunks.

        Args:
            text: Text to synthesize
            yield_timing: If True, yield (audio, timings) tuples

        Yields:
            Float32 audio chunks normalized to [-1, 1], or (audio, timings) tuples
        """
        self.load()

        # Check and clear memory if needed
        if self._check_memory():
            self._clear_memory()

        pipeline = self._get_pipeline(self._current_lang)

        # Determine voice argument (single voice or blend)
        voice_arg: str | torch.Tensor = self._current_voice
        if self._is_blend_formula(self._current_voice):
            voice_arg = self._parse_voice_formula(self._current_voice, pipeline)

        # Generate audio chunks with OOM retry
        try:
            yield from self._generate_chunks(
                text, pipeline, voice_arg, yield_timing
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self._device == "cuda":  # pragma: no cover
                self._clear_memory()
                # Retry once
                yield from self._generate_chunks(
                    text, pipeline, voice_arg, yield_timing
                )
            else:
                raise

    def _generate_chunks(
        self,
        text: str,
        pipeline: KPipeline,
        voice_arg: str | torch.Tensor,
        yield_timing: bool,
    ) -> Iterator[np.ndarray | tuple[np.ndarray, list[TimingInfo]]]:
        """Generate audio chunks from pipeline."""
        audio_duration = 0.0
        generator = pipeline(text, voice=voice_arg, speed=self.config.speed)

        for result in generator:
            audio_float32: np.ndarray = result.audio.cpu().numpy()

            # Apply silence trimming if configured
            if self.config.trim_silence:
                audio_float32 = self._trim_silence(audio_float32)

            # Extract timing info if available and requested
            timings: list[TimingInfo] = []
            if yield_timing and hasattr(result, "tokens") and result.tokens:
                for token in result.tokens:
                    if (
                        token
                        and hasattr(token, "start_ts")
                        and token.start_ts is not None
                        and hasattr(token, "end_ts")
                        and token.end_ts is not None
                        and hasattr(token, "text")
                        and token.text is not None
                    ):
                        timings.append(
                            TimingInfo(
                                start_time=token.start_ts + audio_duration,
                                end_time=token.end_ts + audio_duration,
                                word=token.text,
                            )
                        )

            audio_duration += len(audio_float32) / self.config.sample_rate

            if yield_timing:
                yield audio_float32, timings
            else:
                yield audio_float32

    def synthesize_to_bytes(self, text: str) -> Iterator[bytes]:
        """
        Synthesize text to PCM16 audio bytes.

        Args:
            text: Text to synthesize

        Yields:
            PCM16 audio chunks as bytes
        """
        for result in self.synthesize(text):
            # synthesize() without yield_timing always returns ndarray
            audio_float32: np.ndarray = result  # type: ignore[assignment]
            audio_int16 = (audio_float32 * 32767).astype(np.int16)
            yield audio_int16.tobytes()

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim silence from audio and apply fades."""
        audio = self._trim_silence_start(audio)
        audio = self._trim_silence_end(audio)
        return audio

    def _trim_silence_start(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading silence and apply fade-in."""
        audio = audio.copy()
        trimmed = False

        non_silent = np.where(np.abs(audio) > self.config.silence_threshold)[0]
        if len(non_silent) > 0:
            start_index = non_silent[0]
            if start_index > 0:
                trimmed = True
            audio = audio[start_index:]

        extra_samples = int(
            self.config.extra_trim_start_ms * self.config.sample_rate / 1000
        )
        if extra_samples > 0 and len(audio) > extra_samples:
            audio = audio[extra_samples:]
            trimmed = True

        if trimmed:
            audio = self._apply_fade_in(audio)

        return audio

    def _trim_silence_end(self, audio: np.ndarray) -> np.ndarray:
        """Trim trailing silence and apply fade-out."""
        audio = audio.copy()
        trimmed = False

        non_silent = np.where(np.abs(audio) > self.config.silence_threshold)[0]
        if len(non_silent) > 0:
            end_index = non_silent[-1] + 1
            if end_index < len(audio):
                trimmed = True
            audio = audio[:end_index]

        extra_samples = int(
            self.config.extra_trim_end_ms * self.config.sample_rate / 1000
        )
        if extra_samples > 0 and len(audio) > extra_samples:
            audio = audio[:-extra_samples]
            trimmed = True

        if trimmed:
            audio = self._apply_fade_out(audio)

        return audio

    def _apply_fade_in(self, audio: np.ndarray) -> np.ndarray:
        """Apply linear fade-in to audio."""
        audio = audio.copy()
        fade_samples = int(self.config.fade_in_ms * self.config.sample_rate / 1000)
        if fade_samples == 0 or len(audio) < fade_samples:
            fade_samples = len(audio)
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        audio[:fade_samples] *= fade_in
        return audio

    def _apply_fade_out(self, audio: np.ndarray) -> np.ndarray:
        """Apply linear fade-out to audio."""
        audio = audio.copy()
        fade_samples = int(self.config.fade_out_ms * self.config.sample_rate / 1000)
        if fade_samples == 0 or len(audio) < fade_samples:
            fade_samples = len(audio)
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        audio[-fade_samples:] *= fade_out
        return audio

    def set_voice(self, voice: str) -> None:
        """
        Change the current voice.

        Args:
            voice: Voice name or blend formula
        """
        self._current_voice = voice
        self._current_lang = get_lang_code(voice)

    def get_voices(self) -> list[str]:
        """Get list of available voice names."""
        from .config import KOKORO_VOICES

        return list(KOKORO_VOICES)

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

    def shutdown(self) -> None:
        """Release resources and clear GPU memory."""
        self._pipelines.clear()
        self._blended_voices.clear()
        self._model = None
        self._loaded = False
        self._clear_memory()
