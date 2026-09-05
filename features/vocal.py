"""Vocal / speech impairment feature.

The user types what they want to say - full sentences or just keywords - picks
how it should come out, picks the language it should be spoken in, and
ContextBridge speaks it for them in a natural neural voice.

Pipeline:

    typed text
        -> (optional) Azure AI Foundry rewrite, meaning preserved
        -> (optional) Azure AI Foundry translation into the chosen language
        -> Azure AI Speech synthesis with a neural voice for that language

The pure pipeline lives in ``prepare_message()`` so it can be tested and reused
without Streamlit; ``render()`` is only the interface on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from services import foundry, speech

STATE_PREFIX = "vocal_"

# Short, high-frequency phrases. One tap fills the box for someone who cannot
# type quickly in an urgent moment.
QUICK_PHRASES: tuple[tuple[str, str], ...] = (
    ("🆘 I need help", "I need help, please."),
    ("💊 Medication", "I need to take my medication."),
    ("🚻 Bathroom", "I need to go to the bathroom, please."),
    ("💧 Water", "Could I have a glass of water, please?"),
    ("🩺 In pain", "I am in pain and I need to see someone."),
    ("🕐 One moment", "Please give me a moment, I am typing my answer."),
    ("🙏 Thank you", "Thank you very much for your patience."),
    ("❓ Repeat that", "Could you repeat that more slowly, please?"),
)

MAX_CHARS = 2000
DEFAULT_STYLE = "(default)"


@dataclass
class SpokenMessage:
    """Everything the interface needs to show, plus the text that gets spoken."""

    original: str
    transformed: str
    translated: Optional[str]
    mode: foundry.Mode
    language: speech.Language
    voice: speech.Voice

    @property
    def final_text(self) -> str:
        """The exact text handed to the speech synthesizer."""
        return self.translated or self.transformed

    @property
    def was_rewritten(self) -> bool:
        return self.mode.key != "exact" and self.transformed != self.original

    @property
    def was_translated(self) -> bool:
        return bool(self.translated)


def prepare_message(
    text: str,
    mode_key: str,
    language_code: str,
    voice_id: str,
    *,
    translate: bool = False,
) -> SpokenMessage:
    """Run the text through the Foundry helper and return every stage of it.

    Raises ``foundry.FoundryError`` if a model call fails; the caller decides
    how to recover.
    """
    original = (text or "").strip()
    if not original:
        raise ValueError("There is no text to prepare.")

    mode = foundry.get_mode(mode_key)
    language = speech.get_language(language_code)
    voice = speech.find_voice(voice_id) or speech.default_voice(language_code)

    transformed = foundry.transform_text(original, mode.key)

    translated = None
    if translate:
        translated = foundry.translate_text(
            transformed,
            language.translate_name,
            tone_hint=mode.label.lower() if mode.key in ("polite", "formal") else None,
        )

    return SpokenMessage(
        original=original,
        transformed=transformed,
        translated=translated,
        mode=mode,
        language=language,
        voice=voice,
    )


# --- Interface ---------------------------------------------------------------


def _state(key: str, default):
    full = STATE_PREFIX + key
    if full not in st.session_state:
        st.session_state[full] = default
    return st.session_state[full]


def _set(key: str, value) -> None:
    st.session_state[STATE_PREFIX + key] = value


def _get(key: str, default=None):
    return st.session_state.get(STATE_PREFIX + key, default)


def render() -> None:
    st.header("🗣️ Vocal / speech impairment")
    st.caption(
        "Type what you want to say - or just the keywords. ContextBridge turns it "
        "into a natural sentence and speaks it out loud in the language you choose. "
        "Your meaning is never changed."
    )

    _state("text", "")
    _state("message", None)
    _state("audio", None)
    _state("history", [])

    _render_composer()
    st.divider()
    _render_result()
    _render_history()


def _render_composer() -> None:
    st.subheader("1. What do you want to say?")

    st.caption("Quick phrases")
    columns = st.columns(4)
    for index, (label, phrase) in enumerate(QUICK_PHRASES):
        if columns[index % 4].button(label, key="vocal_quick_" + str(index), use_container_width=True):
            _set("text", phrase)
            st.rerun()

    st.text_area(
        "Your text or keywords",
        key=STATE_PREFIX + "text",
        height=120,
        max_chars=MAX_CHARS,
        placeholder="water cold please\n...or a full sentence, whichever is easier.",
        label_visibility="collapsed",
    )

    st.subheader("2. How should it come out?")
    mode_key = st.radio(
        "Transformation",
        options=[mode.key for mode in foundry.MODES],
        format_func=lambda key: foundry.MODES_BY_KEY[key].label,
        key=STATE_PREFIX + "mode",
        label_visibility="collapsed",
    )
    st.caption(foundry.MODES_BY_KEY[mode_key].help)

    st.subheader("3. Which voice should say it?")
    left, right = st.columns(2)

    languages = speech.list_languages()
    default_index = next(
        (i for i, lang in enumerate(languages) if lang.code == speech.DEFAULT_TTS_LANGUAGE),
        0,
    )
    language_code = left.selectbox(
        "Spoken language",
        options=[lang.code for lang in languages],
        format_func=lambda code: speech.get_language(code).label,
        index=default_index,
        key=STATE_PREFIX + "language",
    )
    language = speech.get_language(language_code)

    voices = speech.list_voices(language_code)
    voice_id = right.selectbox(
        "Voice",
        options=[voice.id for voice in voices],
        format_func=lambda vid: (speech.find_voice(vid) or voices[0]).label,
        key=STATE_PREFIX + "voice_" + language_code,
    )

    translate = st.toggle(
        "Translate my message into " + language.name,
        value=False,
        key=STATE_PREFIX + "translate",
        help=(
            "Leave this off if you already typed in " + language.name + ". "
            "Turn it on to type in your own language and be heard in this one."
        ),
    )

    with st.expander("Voice settings"):
        st.slider(
            "Speaking speed",
            -40, 40, 0, 5,
            key=STATE_PREFIX + "rate",
            help="Slower can be easier for a listener to follow.",
        )
        st.slider("Pitch", -30, 30, 0, 5, key=STATE_PREFIX + "pitch")

        voice = speech.find_voice(voice_id)
        chosen_style = None
        if voice and voice.styles:
            # The widget owns "style_choice"; the resolved value lives in "style",
            # because Streamlit forbids writing back to a live widget's own key.
            picked = st.selectbox(
                "Speaking style",
                options=(DEFAULT_STYLE,) + voice.styles,
                key=STATE_PREFIX + "style_choice",
            )
            chosen_style = None if picked == DEFAULT_STYLE else picked
        _set("style", chosen_style)

    text = (_get("text") or "").strip()
    needs_model = mode_key != "exact" or translate

    prepare_label = "✨ Prepare my message" if needs_model else "✅ Use my text as it is"
    if st.button(
        prepare_label,
        type="primary",
        disabled=not text,
        use_container_width=True,
        key=STATE_PREFIX + "prepare",
    ):
        _prepare(text, mode_key, language_code, voice_id, translate)

    foundry_ok, foundry_message = foundry.is_configured()
    if needs_model and not foundry_ok:
        st.warning(
            foundry_message + " Rewriting and translation are unavailable, but "
            "you can still use **Speak exactly what I typed**.",
            icon="⚠️",
        )


def _prepare(text: str, mode_key: str, language_code: str, voice_id: str, translate: bool) -> None:
    if translate:
        spinner = "Rewriting and translating your message..."
    elif mode_key != "exact":
        spinner = "Rewriting your message..."
    else:
        spinner = "Getting your message ready..."
    try:
        with st.spinner(spinner):
            message = prepare_message(
                text, mode_key, language_code, voice_id, translate=translate
            )
    except foundry.FoundryNotConfigured as exc:
        st.error(str(exc), icon="🔌")
        return
    except foundry.FoundryError as exc:
        st.error(
            "Your message could not be prepared, so nothing was changed. " + str(exc),
            icon="⚠️",
        )
        st.info("You can still speak your original words with **Speak exactly what I typed**.")
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    _set("message", message)
    _set("audio", None)


def _render_result() -> None:
    message: Optional[SpokenMessage] = _get("message")
    if message is None:
        st.info("Type something above and press **Prepare my message** to see it here.", icon="💬")
        return

    st.subheader("Your message")

    left, right = st.columns(2)
    with left:
        st.markdown("**Original - what you typed**")
        st.info(message.original)
    with right:
        if message.mode.key == "exact":
            st.markdown("**Transformed**")
            st.info("Not rewritten - your exact words are used.")
        else:
            st.markdown("**Transformed - " + message.mode.label.lower() + "**")
            st.success(message.transformed)

    if message.was_translated:
        st.markdown("**Translated - " + message.language.label + "**")
        st.success(message.translated)

    st.markdown("**🔊 This is what will be spoken**")
    try:
        st.code(message.final_text, language=None, wrap_lines=True)
    except TypeError:  # wrap_lines landed in a later Streamlit
        st.code(message.final_text, language=None)
    st.caption(
        "Voice: " + message.voice.label + " · " + message.language.label +
        " · Meaning is preserved; only the wording changes."
    )

    # speech.is_configured() returns (ok, message) - never test the tuple itself,
    # a non-empty tuple is always truthy.
    speech_ok, speech_message = speech.is_configured()
    if not speech_ok:
        st.warning(
            speech_message + " Playback is unavailable until this is fixed.",
            icon="🔌",
        )
        return

    speak, edit = st.columns([3, 1])
    if speak.button(
        "🔊 Speak this out loud",
        type="primary",
        use_container_width=True,
        key=STATE_PREFIX + "speak",
    ):
        _speak(message)
    if edit.button("✏️ Start over", use_container_width=True, key=STATE_PREFIX + "reset"):
        _set("message", None)
        _set("audio", None)
        st.rerun()

    audio = _get("audio")
    if audio:
        _play(audio, autoplay=True)
        st.download_button(
            "⬇️ Save this audio",
            data=audio,
            file_name="contextbridge-message.mp3",
            mime="audio/mpeg",
            key=STATE_PREFIX + "download",
        )


def _speak(message: SpokenMessage) -> None:
    try:
        with st.spinner("Speaking with " + message.voice.name + "..."):
            audio = speech.synthesize(
                message.final_text,
                message.voice.id,
                locale=message.language.code,
                rate=int(_get("rate", 0) or 0),
                pitch=int(_get("pitch", 0) or 0),
                style=_get("style"),
            )
    except speech.SpeechNotConfigured as exc:
        st.error(str(exc), icon="🔌")
        return
    except speech.SpeechError as exc:
        st.error("The voice could not be generated. " + str(exc), icon="⚠️")
        return

    _set("audio", audio)
    history = list(_get("history", []))
    history.insert(0, {"text": message.final_text, "voice": message.voice.label, "audio": audio})
    _set("history", history[:5])


def _play(audio: bytes, *, autoplay: bool = False) -> None:
    """st.audio gained `autoplay` in Streamlit 1.38; degrade gracefully."""
    try:
        st.audio(audio, format="audio/mp3", autoplay=autoplay)
    except TypeError:
        st.audio(audio, format="audio/mp3")


def _render_history() -> None:
    history = _get("history", [])
    if not history:
        return
    with st.expander("🕘 Recently spoken (" + str(len(history)) + ")"):
        for index, item in enumerate(history):
            st.caption(item["voice"])
            st.write(item["text"])
            _play(item["audio"])
            if index < len(history) - 1:
                st.divider()


# Allows `streamlit run features/vocal.py` for solo testing of this feature.
if __name__ == "__main__":
    st.set_page_config(page_title="ContextBridge - Vocal", page_icon="🗣️")
    render()
