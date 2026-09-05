"""ContextBridge - Hearing impairment flow.

Records a spoken announcement, transcribes it with Azure AI Speech, then asks
the shared Foundry text helper to turn the transcript into whatever kind of
help the user selected (captions, main message, action, place, deadline,
simplified wording, reduced overload).

Run it on its own with::

    streamlit run features/hearing.py

or mount it in the shared app with ``features.hearing.render()``.
"""

from __future__ import annotations

import html
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

# Allow `streamlit run features/hearing.py` to import the services package.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services import speech  # noqa: E402  (must follow the sys.path bootstrap)

# ---------------------------------------------------------------------------
# Shared Foundry text helper
#
# Contract this module expects from services/foundry.py:
#
#     def generate_text(system_prompt: str, user_prompt: str) -> str: ...
#
# A one-argument helper is also supported: the two prompts are then joined
# before the call. Nothing else in this file touches Foundry directly.
# ---------------------------------------------------------------------------

_FOUNDRY_FUNCTION_NAMES = (
    "generate_text",
    "run_text_task",
    "ask_foundry",
    "complete",
    "chat",
    "ask",
    "run",
)


def _resolve_foundry():
    """Find the shared Foundry helper, tolerating a few likely names.

    Returns ``(callable, error_message)``; exactly one of them is set so the
    UI can explain what the other team member still has to provide.
    """
    try:
        from services import foundry
    except ImportError:
        return None, (
            "services/foundry.py is not available yet. This feature needs a "
            "helper with the signature "
            "`generate_text(system_prompt: str, user_prompt: str) -> str`."
        )

    for name in _FOUNDRY_FUNCTION_NAMES:
        candidate = getattr(foundry, name, None)
        if callable(candidate):
            return candidate, None

    return None, (
        "services/foundry.py was found but exposes none of the expected "
        f"functions ({', '.join(_FOUNDRY_FUNCTION_NAMES)}). Please expose "
        "`generate_text(system_prompt, user_prompt)`."
    )


def _call_foundry(helper, system_prompt: str, user_prompt: str) -> str:
    """Call the shared helper with whichever arity it declares."""
    try:
        parameters = [
            parameter
            for parameter in inspect.signature(helper).parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        takes_two = len(parameters) >= 2
    except (TypeError, ValueError):  # builtins / C callables
        takes_two = True

    if takes_two:
        return str(helper(system_prompt, user_prompt))
    return str(helper(f"{system_prompt}\n\n---\n\n{user_prompt}"))


# ---------------------------------------------------------------------------
# Assistance options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssistanceOption:
    """One kind of help the user can ask for."""

    key: str
    label: str
    help_text: str
    heading: str
    icon: str
    # JSON field requested from Foundry; None means the app answers locally.
    field: str | None = None
    instruction: str | None = None
    is_list: bool = False


ASSISTANCE_OPTIONS: tuple[AssistanceOption, ...] = (
    AssistanceOption(
        key="captions",
        label="Show full captions / transcript",
        help_text="The complete text of what was said, with timings.",
        heading="Full transcript",
        icon="[CC]",
    ),
    AssistanceOption(
        key="main_message",
        label="Extract main message",
        help_text="One or two sentences with the point of the announcement.",
        heading="Main message",
        icon="[!]",
        field="main_message",
        instruction=(
            "main_message: the single most important thing the announcement "
            "says, in at most two short sentences."
        ),
    ),
    AssistanceOption(
        key="action",
        label="Highlight required action",
        help_text="What the listener is expected to do.",
        heading="What you have to do",
        icon="[>]",
        field="action_required",
        instruction=(
            "action_required: what the listener is asked to do. If the "
            'announcement asks for nothing, return exactly "Not mentioned".'
        ),
    ),
    AssistanceOption(
        key="location",
        label="Highlight location",
        help_text="The place named in the announcement.",
        heading="Where",
        icon="[@]",
        field="location",
        instruction=(
            "location: the place named in the announcement (gate, platform, "
            'room, address, building). If no place is named, return exactly "Not mentioned".'
        ),
    ),
    AssistanceOption(
        key="deadline",
        label="Highlight time / deadline",
        help_text="When it happens or by when to act.",
        heading="When",
        icon="[T]",
        field="time_deadline",
        instruction=(
            "time_deadline: the time, date or deadline stated. If no time is "
            'stated, return exactly "Not mentioned".'
        ),
    ),
    AssistanceOption(
        key="simplify",
        label="Simplify the announcement",
        help_text="The same content in plain, easy language.",
        heading="In simple words",
        icon="[=]",
        field="simplified_announcement",
        instruction=(
            "simplified_announcement: the whole announcement rewritten in "
            "plain language, short sentences, no jargon, keeping every fact."
        ),
    ),
    AssistanceOption(
        key="reduce",
        label="Reduce information overload",
        help_text="Only the few points that actually matter.",
        heading="Only what matters",
        icon="[*]",
        field="key_points",
        instruction=(
            "key_points: a JSON array of at most 4 short strings, each one "
            "fact the listener truly needs. Drop filler, greetings and repetition."
        ),
        is_list=True,
    ),
)

DEFAULT_SELECTED_KEYS = {"captions", "main_message", "action", "deadline"}

NOT_MENTIONED = "Not mentioned"

RESULT_LANGUAGES = {
    "ro-RO": "Romanian",
    "en-US": "English",
    "en-GB": "English",
    "de-DE": "German",
    "fr-FR": "French",
    "es-ES": "Spanish",
    "it-IT": "Italian",
    "hu-HU": "Hungarian",
}


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompts(
    transcript: str, selected: list[AssistanceOption], result_language: str
) -> tuple[str, str]:
    """Build the Foundry system and user prompts for the selected options."""
    tasks = [option.instruction for option in selected if option.instruction]
    fields = [option.field for option in selected if option.field]

    system_prompt = "\n".join(
        [
            "You are ContextBridge, an assistant that makes spoken "
            "announcements accessible to deaf and hard-of-hearing people.",
            "",
            "Absolute rules:",
            "- Use ONLY information that is present in the transcript.",
            "- Never invent, guess, complete or add any detail.",
            f'- If the transcript does not contain a requested detail, return exactly "{NOT_MENTIONED}".',
            "- Do not change meaning, do not soften and do not dramatise.",
            "- Use short, plain sentences. Avoid jargon, idioms and abbreviations.",
            f"- Write every value in {result_language}.",
            "",
            "Produce these fields:",
            *[f"- {task}" for task in tasks],
            "",
            "Answer with a single JSON object and nothing else. It must have "
            f"exactly these keys: {', '.join(fields)}.",
            "Do not wrap the JSON in code fences and do not add commentary.",
        ]
    )

    user_prompt = f"Transcript of the announcement:\n\"\"\"\n{transcript.strip()}\n\"\"\""
    return system_prompt, user_prompt


def parse_foundry_response(raw: str, selected: list[AssistanceOption]) -> tuple[dict, str | None]:
    """Parse the JSON object out of the model reply.

    Returns ``(values, error)``. On a parse failure the raw text is handed
    back so the user still sees something useful instead of an empty screen.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        _, _, text = text.partition("\n")

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}, "The assistant did not return JSON."

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return {}, f"The assistant returned malformed JSON ({exc.msg})."

    if not isinstance(payload, dict):
        return {}, "The assistant did not return a JSON object."

    values = {}
    for option in selected:
        if option.field:
            values[option.field] = payload.get(option.field, NOT_MENTIONED)
    return values, None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_STYLE = """
<style>
  .cb-answer { font-size: 1.15rem; line-height: 1.65; margin: 0.15rem 0 0 0; }
  .cb-missing { font-size: 1.05rem; opacity: 0.65; font-style: italic; margin: 0; }
  .cb-caption-time { font-variant-numeric: tabular-nums; opacity: 0.6;
                     margin-right: 0.6rem; }
</style>
"""


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return str(value).strip().lower() in {"", "not mentioned", "not mentioned."}


def _render_value(value: object, is_list: bool) -> None:
    if _is_missing(value):
        st.markdown(
            f'<p class="cb-missing">{NOT_MENTIONED} in the announcement.</p>',
            unsafe_allow_html=True,
        )
        return

    if is_list and isinstance(value, (list, tuple)):
        for item in value:
            st.markdown(
                f'<p class="cb-answer">&bull; {html.escape(str(item))}</p>',
                unsafe_allow_html=True,
            )
        return

    text = html.escape(str(value)).replace("\n", "<br>")
    st.markdown(f'<p class="cb-answer">{text}</p>', unsafe_allow_html=True)


def _pin_result_language() -> None:
    """Remember that the user chose the result language themselves."""
    st.session_state["hearing_result_language_pinned"] = True


def _build_plain_text(result: speech.TranscriptionResult, selected, values) -> str:
    """Flatten everything on screen into a downloadable text file."""
    lines = ["ContextBridge - accessible announcement", ""]
    for option in selected:
        if option.key == "captions":
            lines += [option.heading.upper(), result.text, ""]
            continue
        value = values.get(option.field)
        if isinstance(value, (list, tuple)):
            rendered = "\n".join(f"- {item}" for item in value) or NOT_MENTIONED
        else:
            rendered = NOT_MENTIONED if _is_missing(value) else str(value)
        lines += [option.heading.upper(), rendered, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Draw the hearing-impairment flow."""
    st.markdown(_STYLE, unsafe_allow_html=True)
    st.title("Hearing support")
    st.write(
        "Record a spoken announcement. ContextBridge writes it down and turns "
        "it into exactly the kind of help you choose."
    )

    speech_ok, speech_message = speech.is_configured()
    if not speech_ok:
        st.error(speech_message)

    settings, options_column = st.columns([1, 1], gap="large")

    with settings:
        st.subheader("Language")
        spoken_label = st.selectbox(
            "Language spoken in the announcement",
            list(speech.SUPPORTED_LANGUAGES.keys()),
            key="hearing_spoken_label",
        )
        spoken_code = speech.SUPPORTED_LANGUAGES[spoken_label]

        # The result language follows the spoken language until the user picks
        # one explicitly; from then on their choice wins.
        if not st.session_state.get("hearing_result_language_pinned"):
            st.session_state["hearing_result_language"] = RESULT_LANGUAGES.get(
                spoken_code, "English"
            )

        result_language = st.selectbox(
            "Language of the result",
            sorted(set(RESULT_LANGUAGES.values())),
            key="hearing_result_language",
            on_change=_pin_result_language,
            help="Follows the spoken language until you change it yourself.",
        )

    with options_column:
        st.subheader("What help do you need?")
        selected_keys = [
            option.key
            for option in ASSISTANCE_OPTIONS
            if st.checkbox(
                option.label,
                value=option.key in DEFAULT_SELECTED_KEYS,
                help=option.help_text,
                key=f"hearing_option_{option.key}",
            )
        ]

    selected = [option for option in ASSISTANCE_OPTIONS if option.key in selected_keys]

    st.divider()
    st.subheader("The announcement")

    recording = st.audio_input("Record the announcement", key="hearing_recording")
    uploaded = st.file_uploader(
        "...or upload an audio file",
        type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
        key="hearing_upload",
        help="Useful when the announcement was captured on another device.",
    )

    source = recording or uploaded
    transcribe_clicked = st.button(
        "Transcribe and help me",
        type="primary",
        disabled=source is None or not speech_ok,
        use_container_width=True,
    )

    if transcribe_clicked and source is not None:
        with st.spinner("Listening to the announcement..."):
            try:
                st.session_state["hearing_result"] = speech.transcribe(
                    source.getvalue(), language=spoken_code
                )
                st.session_state.pop("hearing_assistance", None)
            except speech.SpeechServiceError as exc:
                st.session_state.pop("hearing_result", None)
                st.error(str(exc))

    result: speech.TranscriptionResult | None = st.session_state.get("hearing_result")
    if result is None:
        return

    for warning in result.warnings:
        st.warning(warning)

    if result.is_empty:
        st.warning(
            "No speech was recognised. Check that the right language is "
            "selected and that the recording is not silent."
        )
        return

    st.success(
        f"Transcribed {result.audio_seconds:.0f} seconds of audio "
        f"({len(result.captions)} phrases)."
    )

    if not selected:
        st.info("Select at least one kind of help above.")
        return

    # Ask Foundry only for the fields that actually need the model.
    values: dict = {}
    model_options = [option for option in selected if option.field]
    if model_options:
        cache_key = (result.text, tuple(sorted(o.key for o in model_options)), result_language)
        cached = st.session_state.get("hearing_assistance")

        if cached and cached.get("key") == cache_key:
            values = cached["values"]
        else:
            helper, helper_error = _resolve_foundry()
            if helper_error:
                st.error(helper_error)
            else:
                system_prompt, user_prompt = build_prompts(
                    result.text, model_options, result_language
                )
                with st.spinner("Preparing your accessible summary..."):
                    try:
                        raw = _call_foundry(helper, system_prompt, user_prompt)
                    except Exception as exc:  # noqa: BLE001 - surfaced to the user
                        st.error(f"The Foundry text helper failed: {exc}")
                        raw = ""

                if raw:
                    values, parse_error = parse_foundry_response(raw, model_options)
                    if parse_error:
                        st.warning(f"{parse_error} Showing the raw answer instead.")
                        st.write(raw)
                    else:
                        st.session_state["hearing_assistance"] = {
                            "key": cache_key,
                            "values": values,
                        }

    st.divider()
    for option in selected:
        with st.container(border=True):
            st.markdown(f"#### {option.icon} {option.heading}")
            if option.key == "captions":
                for caption in result.captions:
                    st.markdown(
                        f'<p class="cb-answer">'
                        f'<span class="cb-caption-time">{caption.timestamp}</span>'
                        f"{html.escape(caption.text)}</p>",
                        unsafe_allow_html=True,
                    )
                with st.expander("Copy the plain transcript"):
                    st.code(result.text, language=None)
            else:
                _render_value(values.get(option.field), option.is_list)

    st.download_button(
        "Download this summary",
        data=_build_plain_text(result, selected, values),
        file_name="contextbridge-announcement.txt",
        mime="text/plain",
        use_container_width=True,
    )


if __name__ == "__main__":
    st.set_page_config(page_title="ContextBridge - Hearing support", page_icon="CC")
    render()
