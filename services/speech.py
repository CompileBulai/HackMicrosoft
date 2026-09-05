"""Azure AI Speech integration for ContextBridge.

Every call into the Azure Speech SDK lives in this module. Feature modules
import the small surface below and never touch
``azure.cognitiveservices.speech`` directly.

The module has two halves that share one Azure Speech resource:

* speech-to-text  - ``transcribe()``, used by the hearing flow
* text-to-speech  - ``synthesize()`` and the voice catalogue, used by the
  vocal flow

Environment variables
---------------------
AZURE_SPEECH_KEY      required - Speech resource key
AZURE_SPEECH_REGION   required - e.g. "westeurope", "swedencentral"
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass, field
from xml.sax.saxutils import escape as xml_escape

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


# ===========================================================================
# Text-to-speech
#
# Everything above turns speech into text for the hearing flow. Everything
# below turns text into speech for the vocal flow. Both halves share the same
# Azure Speech resource, the same credentials and the same `is_configured()`.
# ===========================================================================


class SpeechError(SpeechServiceError):
    """Raised when speech could not be synthesised."""


class SpeechNotConfigured(SpeechError):
    """Credentials or the Speech SDK are missing."""


@dataclass(frozen=True)
class Voice:
    """One Azure neural voice."""

    id: str
    name: str
    gender: str
    note: str = ""
    styles: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        bits = [self.gender]
        if self.note:
            bits.append(self.note)
        return f"{self.name} ({', '.join(bits)})"


@dataclass(frozen=True)
class Language:
    """A spoken language offered by the vocal flow."""

    code: str  # Azure locale, e.g. ro-RO
    name: str  # shown in the picker, e.g. Romanian
    translate_name: str  # handed to the Foundry translation prompt
    flag: str
    voices: tuple[Voice, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.flag} {self.name}"


_JENNY_STYLES = (
    "friendly",
    "cheerful",
    "sad",
    "excited",
    "hopeful",
    "shouting",
    "whispering",
)

# Voices marked multilingual are the most natural sounding and handle foreign
# names inside a sentence gracefully.
LANGUAGES: tuple[Language, ...] = (
    Language(
        "en-US", "English (United States)", "English", "\U0001F1FA\U0001F1F8",
        (
            Voice("en-US-AvaMultilingualNeural", "Ava", "Female", "multilingual, very natural"),
            Voice("en-US-AndrewMultilingualNeural", "Andrew", "Male", "multilingual, warm"),
            Voice("en-US-EmmaMultilingualNeural", "Emma", "Female", "multilingual, bright"),
            Voice("en-US-BrianMultilingualNeural", "Brian", "Male", "multilingual, calm"),
            Voice("en-US-JennyNeural", "Jenny", "Female", "expressive", _JENNY_STYLES),
        ),
    ),
    Language(
        "en-GB", "English (United Kingdom)", "British English", "\U0001F1EC\U0001F1E7",
        (
            Voice("en-GB-SoniaNeural", "Sonia", "Female", "clear"),
            Voice("en-GB-RyanNeural", "Ryan", "Male", "warm"),
            Voice("en-GB-LibbyNeural", "Libby", "Female", "friendly"),
        ),
    ),
    Language(
        "ro-RO", "Romanian", "Romanian", "\U0001F1F7\U0001F1F4",
        (
            Voice("ro-RO-AlinaNeural", "Alina", "Female", "natural"),
            Voice("ro-RO-EmilNeural", "Emil", "Male", "natural"),
        ),
    ),
    Language(
        "es-ES", "Spanish (Spain)", "European Spanish", "\U0001F1EA\U0001F1F8",
        (
            Voice("es-ES-ElviraNeural", "Elvira", "Female", "natural"),
            Voice("es-ES-AlvaroNeural", "Alvaro", "Male", "natural"),
            Voice("es-ES-XimenaNeural", "Ximena", "Female", "soft"),
        ),
    ),
    Language(
        "es-MX", "Spanish (Mexico)", "Latin American Spanish", "\U0001F1F2\U0001F1FD",
        (
            Voice("es-MX-DaliaNeural", "Dalia", "Female", "natural"),
            Voice("es-MX-JorgeNeural", "Jorge", "Male", "natural"),
        ),
    ),
    Language(
        "fr-FR", "French", "French", "\U0001F1EB\U0001F1F7",
        (
            Voice("fr-FR-VivienneMultilingualNeural", "Vivienne", "Female", "multilingual"),
            Voice("fr-FR-DeniseNeural", "Denise", "Female", "natural"),
            Voice("fr-FR-HenriNeural", "Henri", "Male", "natural"),
        ),
    ),
    Language(
        "de-DE", "German", "German", "\U0001F1E9\U0001F1EA",
        (
            Voice("de-DE-SeraphinaMultilingualNeural", "Seraphina", "Female", "multilingual"),
            Voice("de-DE-KatjaNeural", "Katja", "Female", "natural"),
            Voice("de-DE-ConradNeural", "Conrad", "Male", "natural"),
        ),
    ),
    Language(
        "it-IT", "Italian", "Italian", "\U0001F1EE\U0001F1F9",
        (
            Voice("it-IT-ElsaNeural", "Elsa", "Female", "natural"),
            Voice("it-IT-IsabellaNeural", "Isabella", "Female", "expressive"),
            Voice("it-IT-DiegoNeural", "Diego", "Male", "natural"),
        ),
    ),
    Language(
        "pt-BR", "Portuguese (Brazil)", "Brazilian Portuguese", "\U0001F1E7\U0001F1F7",
        (
            Voice("pt-BR-ThalitaMultilingualNeural", "Thalita", "Female", "multilingual"),
            Voice("pt-BR-FranciscaNeural", "Francisca", "Female", "natural"),
            Voice("pt-BR-AntonioNeural", "Antonio", "Male", "natural"),
        ),
    ),
    Language(
        "nl-NL", "Dutch", "Dutch", "\U0001F1F3\U0001F1F1",
        (
            Voice("nl-NL-FennaNeural", "Fenna", "Female", "natural"),
            Voice("nl-NL-MaartenNeural", "Maarten", "Male", "natural"),
        ),
    ),
    Language(
        "pl-PL", "Polish", "Polish", "\U0001F1F5\U0001F1F1",
        (
            Voice("pl-PL-ZofiaNeural", "Zofia", "Female", "natural"),
            Voice("pl-PL-MarekNeural", "Marek", "Male", "natural"),
        ),
    ),
    Language(
        "hu-HU", "Hungarian", "Hungarian", "\U0001F1ED\U0001F1FA",
        (
            Voice("hu-HU-NoemiNeural", "Noemi", "Female", "natural"),
            Voice("hu-HU-TamasNeural", "Tamas", "Male", "natural"),
        ),
    ),
    Language(
        "uk-UA", "Ukrainian", "Ukrainian", "\U0001F1FA\U0001F1E6",
        (
            Voice("uk-UA-PolinaNeural", "Polina", "Female", "natural"),
            Voice("uk-UA-OstapNeural", "Ostap", "Male", "natural"),
        ),
    ),
    Language(
        "tr-TR", "Turkish", "Turkish", "\U0001F1F9\U0001F1F7",
        (
            Voice("tr-TR-EmelNeural", "Emel", "Female", "natural"),
            Voice("tr-TR-AhmetNeural", "Ahmet", "Male", "natural"),
        ),
    ),
    Language(
        "ar-EG", "Arabic (Egypt)", "Egyptian Arabic", "\U0001F1EA\U0001F1EC",
        (
            Voice("ar-EG-SalmaNeural", "Salma", "Female", "natural"),
            Voice("ar-EG-ShakirNeural", "Shakir", "Male", "natural"),
        ),
    ),
    Language(
        "hi-IN", "Hindi", "Hindi", "\U0001F1EE\U0001F1F3",
        (
            Voice("hi-IN-SwaraNeural", "Swara", "Female", "natural"),
            Voice("hi-IN-MadhurNeural", "Madhur", "Male", "natural"),
        ),
    ),
    Language(
        "ja-JP", "Japanese", "Japanese", "\U0001F1EF\U0001F1F5",
        (
            Voice("ja-JP-NanamiNeural", "Nanami", "Female", "natural"),
            Voice("ja-JP-KeitaNeural", "Keita", "Male", "natural"),
        ),
    ),
    Language(
        "ko-KR", "Korean", "Korean", "\U0001F1F0\U0001F1F7",
        (
            Voice("ko-KR-SunHiNeural", "Sun-Hi", "Female", "natural"),
            Voice("ko-KR-InJoonNeural", "In-Joon", "Male", "natural"),
        ),
    ),
    Language(
        "zh-CN", "Chinese (Mandarin)", "Simplified Chinese (Mandarin)", "\U0001F1E8\U0001F1F3",
        (
            Voice("zh-CN-XiaoxiaoMultilingualNeural", "Xiaoxiao", "Female", "multilingual"),
            Voice("zh-CN-YunxiNeural", "Yunxi", "Male", "natural"),
        ),
    ),
)

LANGUAGES_BY_CODE: dict[str, Language] = {language.code: language for language in LANGUAGES}

# The vocal flow's own default. Kept separate from DEFAULT_LANGUAGE above,
# which is the language the hearing flow expects to *hear*.
DEFAULT_TTS_LANGUAGE = "en-US"


def list_languages() -> tuple[Language, ...]:
    return LANGUAGES


def get_language(code: str) -> Language:
    try:
        return LANGUAGES_BY_CODE[code]
    except KeyError:
        raise SpeechError(f"Unknown language code: {code!r}") from None


def list_voices(language_code: str) -> tuple[Voice, ...]:
    return get_language(language_code).voices


def default_voice(language_code: str) -> Voice:
    return get_language(language_code).voices[0]


def find_voice(voice_id: str) -> Voice | None:
    for language in LANGUAGES:
        for voice in language.voices:
            if voice.id == voice_id:
                return voice
    return None


def locale_of(voice_id: str) -> str:
    """Derive the locale from a voice id, e.g. ro-RO-AlinaNeural -> ro-RO."""
    match = re.match(r"^([a-z]{2,3}-[A-Za-z]{2,8})-", voice_id)
    return match.group(1) if match else DEFAULT_TTS_LANGUAGE


def status() -> dict:
    """Configuration state in the shape the app shell's status panel wants."""
    ok, message = is_configured()
    return {
        "configured": ok,
        "message": message,
        "region": os.getenv("AZURE_SPEECH_REGION") or "(not set)",
        "key_set": bool(os.getenv("AZURE_SPEECH_KEY")),
    }


def get_speech_config(voice_id: str | None = None):
    """Build a SpeechConfig for synthesis, in MP3 so a browser can play it."""
    ok, message = is_configured()
    if not ok:
        raise SpeechNotConfigured(message)

    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    if voice_id:
        speech_config.speech_synthesis_voice_name = voice_id
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio24Khz96KBitRateMonoMp3
    )
    return speech_config


def _percent(value: int) -> str:
    value = int(value)
    return f"{'+' if value >= 0 else ''}{value}%"


def build_ssml(
    text: str,
    voice_id: str,
    *,
    locale: str | None = None,
    rate: int = 0,
    pitch: int = 0,
    style: str | None = None,
) -> str:
    """Wrap text in SSML so we control locale, speed and pitch precisely."""
    locale = locale or locale_of(voice_id)
    body = xml_escape(text.strip())
    body = f'<prosody rate="{_percent(rate)}" pitch="{_percent(pitch)}">{body}</prosody>'
    if style:
        body = f'<mstts:express-as style="{xml_escape(style)}">{body}</mstts:express-as>'
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{locale}">'
        f'<voice name="{voice_id}">{body}</voice></speak>'
    )


def synthesize(
    text: str,
    voice_id: str,
    *,
    locale: str | None = None,
    rate: int = 0,
    pitch: int = 0,
    style: str | None = None,
) -> bytes:
    """Speak ``text`` with ``voice_id`` and return MP3 bytes.

    Nothing is played on the server's own speakers: ``audio_config=None`` keeps
    the audio in memory so the caller can hand it to ``st.audio``.
    """
    text = (text or "").strip()
    if not text:
        raise SpeechError("There is no text to speak.")

    speech_config = get_speech_config(voice_id)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    ssml = build_ssml(text, voice_id, locale=locale, rate=rate, pitch=pitch, style=style)
    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return bytes(result.audio_data)

    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        message = f"Azure Speech cancelled the request: {details.reason}"
        if getattr(details, "error_details", None):
            message += f" - {details.error_details}"
        raise SpeechError(message)

    raise SpeechError(f"Speech synthesis failed: {result.reason}")
