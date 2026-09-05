"""Azure AI Speech integration for ContextBridge.

Every call into the Azure Speech SDK lives in this module. Feature modules
import the small surface below and never touch
``azure.cognitiveservices.speech`` directly.

Environment variables
---------------------
AZURE_SPEECH_KEY      required - Speech resource key
AZURE_SPEECH_REGION   required - e.g. "westeurope", "swedencentral"
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass, field

try:  # optional: keeps the module importable before `pip install`
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:  # pragma: no cover - surfaced as a friendly UI error
    speechsdk = None


# Languages offered in the UI: label -> BCP-47 code understood by Azure Speech.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "Romanian (ro-RO)": "ro-RO",
    "English (en-US)": "en-US",
    "English (en-GB)": "en-GB",
    "German (de-DE)": "de-DE",
    "French (fr-FR)": "fr-FR",
    "Spanish (es-ES)": "es-ES",
    "Italian (it-IT)": "it-IT",
    "Hungarian (hu-HU)": "hu-HU",
}

DEFAULT_LANGUAGE = "ro-RO"

# Continuous recognition on a finished file runs faster than real time; this is
# only a safety net so the UI can never hang forever.
_RECOGNITION_TIMEOUT_SECONDS = 180


class SpeechServiceError(RuntimeError):
    """Raised when transcription cannot be completed."""


@dataclass
class Caption:
    """One recognised phrase, used to build a caption track."""

    text: str
    start_seconds: float
    end_seconds: float

    @property
    def timestamp(self) -> str:
        minutes, seconds = divmod(int(self.start_seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"


@dataclass
class TranscriptionResult:
    """Everything the hearing feature needs after speech-to-text."""

    text: str
    language: str
    captions: list[Caption] = field(default_factory=list)
    audio_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def is_configured() -> tuple[bool, str]:
    """Report whether the Speech SDK and credentials are ready.

    Returns ``(ok, message)`` so the UI can explain exactly what is missing
    instead of failing on the first API call.
    """
    if speechsdk is None:
        return False, (
            "The Azure Speech SDK is not installed. Run: "
            "pip install azure-cognitiveservices-speech"
        )
    if not os.getenv("AZURE_SPEECH_KEY"):
        return False, "AZURE_SPEECH_KEY is not set (put it in your .env file)."
    if not os.getenv("AZURE_SPEECH_REGION"):
        return False, "AZURE_SPEECH_REGION is not set (e.g. westeurope)."
    return True, "Azure Speech is configured."


def _ticks_to_seconds(ticks: int | None) -> float:
    """Azure reports offsets in 100-nanosecond ticks."""
    return (ticks or 0) / 10_000_000


def _wav_duration_seconds(path: str) -> float:
    try:
        with wave.open(path, "rb") as handle:
            return handle.getnframes() / float(handle.getframerate() or 1)
    except Exception:  # noqa: BLE001 - duration is cosmetic, never fatal
        return 0.0


def _convert_to_wav(audio_bytes: bytes, work_dir: str) -> tuple[str, list[str]]:
    """Normalise arbitrary recorded audio to 16 kHz mono 16-bit PCM WAV.

    Browsers hand Streamlit whatever their MediaRecorder produced, while the
    Speech SDK only reads uncompressed WAV without extra codecs installed, so
    we always push the bytes through ffmpeg when it is available.
    """
    warnings: list[str] = []
    source_path = os.path.join(work_dir, "recording.input")
    target_path = os.path.join(work_dir, "recording_16k_mono.wav")

    with open(source_path, "wb") as handle:
        handle.write(audio_bytes)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        warnings.append(
            "ffmpeg was not found, so the recording is sent to Azure as-is. "
            "Install ffmpeg if transcription fails."
        )
        os.replace(source_path, target_path)
        return target_path, warnings

    try:
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", source_path,
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                target_path,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip()
        raise SpeechServiceError(
            f"The recording could not be decoded into WAV audio. {detail}"
        ) from exc

    return target_path, warnings


def _build_speech_config(language: str):
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        raise SpeechServiceError(
            "Azure Speech credentials are missing. Set AZURE_SPEECH_KEY and "
            "AZURE_SPEECH_REGION in your .env file."
        )

    config = speechsdk.SpeechConfig(subscription=key, region=region)
    config.speech_recognition_language = language
    # Announcements often contain a pause before the important part; give the
    # recogniser room instead of cutting the phrase short.
    config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1500"
    )
    return config


def transcribe(audio_bytes: bytes, language: str = DEFAULT_LANGUAGE) -> TranscriptionResult:
    """Transcribe a recorded announcement with Azure AI Speech.

    Uses continuous recognition so announcements longer than a single phrase
    are captured in full, and returns per-phrase captions alongside the joined
    transcript.
    """
    if speechsdk is None:
        raise SpeechServiceError(
            "The Azure Speech SDK is not installed. Run: "
            "pip install azure-cognitiveservices-speech"
        )
    if not audio_bytes:
        raise SpeechServiceError("No audio was recorded.")

    work_dir = tempfile.mkdtemp(prefix="contextbridge_speech_")
    try:
        wav_path, warnings = _convert_to_wav(audio_bytes, work_dir)
        audio_seconds = _wav_duration_seconds(wav_path)

        speech_config = _build_speech_config(language)
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        captions: list[Caption] = []
        errors: list[str] = []
        finished = threading.Event()

        def on_recognized(evt) -> None:
            result = evt.result
            if result.reason != speechsdk.ResultReason.RecognizedSpeech:
                return
            text = (result.text or "").strip()
            if not text:
                return
            start = _ticks_to_seconds(getattr(result, "offset", 0))
            duration = _ticks_to_seconds(getattr(result, "duration", 0))
            captions.append(
                Caption(text=text, start_seconds=start, end_seconds=start + duration)
            )

        def on_canceled(evt) -> None:
            details = getattr(evt, "cancellation_details", None)
            reason = getattr(details, "reason", None) or getattr(evt, "reason", None)
            if reason != speechsdk.CancellationReason.EndOfStream:
                message = (
                    getattr(details, "error_details", None)
                    or getattr(evt, "error_details", None)
                    or "Recognition was cancelled by the service."
                )
                errors.append(str(message))
            finished.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_stopped.connect(lambda _evt: finished.set())

        recognizer.start_continuous_recognition()
        completed = finished.wait(timeout=_RECOGNITION_TIMEOUT_SECONDS)
        recognizer.stop_continuous_recognition()

        if not completed:
            raise SpeechServiceError(
                "Azure Speech did not respond in time. Check your network "
                "connection and try a shorter recording."
            )
        if errors:
            raise SpeechServiceError(f"Azure Speech error: {errors[0]}")

        transcript = " ".join(caption.text for caption in captions).strip()
        return TranscriptionResult(
            text=transcript,
            language=language,
            captions=captions,
            audio_seconds=audio_seconds,
            warnings=warnings,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
