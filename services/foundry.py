"""Shared Azure AI Foundry text helper.

This is the single place in ContextBridge where we talk to a language model.
Every feature (visual, hearing, vocal) should call ``complete()`` or one of the
task helpers below instead of creating its own client.

Design rules that matter for an accessibility tool:
  * The model rewrites, it never invents. Every prompt forbids adding facts.
  * The model answers with the rewritten text only - no preamble, no quotes.
  * Failures raise a typed error so the UI can show a calm message instead of
    a stack trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config


class FoundryError(RuntimeError):
    """Any failure while talking to Azure AI Foundry."""


class FoundryNotConfigured(FoundryError):
    """Credentials are missing from the environment."""


# --- Transformation catalogue ------------------------------------------------
# `key` is what features store, `label` is what the user sees.


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    help: str
    instruction: str


_MEANING_GUARD = (
    "Absolute rules:\n"
    "- Preserve the meaning exactly. Never add facts, names, numbers, dates, "
    "opinions, feelings or requests that are not already in the input.\n"
    "- Never remove information that changes what the person is asking for.\n"
    "- Keep the same point of view: if the person writes in first person, stay "
    "in first person.\n"
    "- Keep the same language as the input unless you are explicitly told to "
    "translate.\n"
    "- The text will be read out loud by a speech synthesizer, so write plain "
    "speakable words: no markdown, no bullet points, no emoji, no quotation "
    "marks around the answer.\n"
    "- Reply with the resulting text only, nothing else."
)

MODES: tuple[Mode, ...] = (
    Mode(
        key="exact",
        label="Speak exactly what I typed",
        help="No AI rewriting at all - your words go straight to the voice.",
        instruction="",
    ),
    Mode(
        key="expand",
        label="Turn keywords into a full sentence",
        help="water cold please -> Could I have some cold water, please?",
        instruction=(
            "The input is a set of keywords or an unfinished fragment typed by "
            "someone who cannot speak easily. Turn it into one natural, complete "
            "sentence that a person would actually say out loud. Add only the "
            "grammatical glue: articles, verbs, pronouns, and politeness that is "
            "already implied by words such as please. Do not invent any new "
            "content and do not answer the sentence."
        ),
    ),
    Mode(
        key="clearer",
        label="Make it clearer",
        help="Same message, easier for a listener to understand first time.",
        instruction=(
            "Rewrite the input so a listener understands it correctly the first "
            "time they hear it. Fix grammar and word order, replace ambiguous "
            "wording with plain wording, and split a tangled sentence into short "
            "ones if that helps. Keep every piece of information."
        ),
    ),
    Mode(
        key="shorter",
        label="Make it shorter",
        help="Trims filler while keeping every piece of information.",
        instruction=(
            "Rewrite the input as briefly as possible while keeping every piece "
            "of information and every request. Remove filler and repetition, not "
            "content. Do not use note-taking style or abbreviations - it still "
            "has to sound like a spoken sentence."
        ),
    ),
    Mode(
        key="polite",
        label="Make it more polite",
        help="Adds warmth and courtesy without changing what you ask for.",
        instruction=(
            "Rewrite the input so it sounds warm, courteous and friendly. Soften "
            "direct commands into requests and add ordinary courtesy words. The "
            "request itself, and how firm it is, must stay exactly the same."
        ),
    ),
    Mode(
        key="formal",
        label="Make it more formal",
        help="Professional register for work, medical or official situations.",
        instruction=(
            "Rewrite the input in a professional, formal register suitable for "
            "work, medical or official situations. Remove slang and contractions "
            "and use complete sentences. Do not make it longer than it needs to "
            "be and do not add flattery."
        ),
    ),
)

MODES_BY_KEY: dict[str, Mode] = {m.key: m for m in MODES}


def get_mode(key: str) -> Mode:
    try:
        return MODES_BY_KEY[key]
    except KeyError:
        raise FoundryError("Unknown transformation mode: " + repr(key)) from None


# --- Client ------------------------------------------------------------------

_client = None


def is_configured() -> tuple[bool, str]:
    """Report whether Foundry is ready, matching services.speech.is_configured.

    Returns ``(ok, message)`` so the UI can say exactly what is missing instead
    of failing on the first API call.
    """
    try:
        import openai  # noqa: F401
    except ImportError:
        return False, (
            "The openai package is not installed. Run: "
            "pip install -r requirements.txt"
        )
    if not config.FOUNDRY_ENDPOINT:
        return False, "AZURE_FOUNDRY_ENDPOINT is not set (put it in your .env file)."
    if not config.FOUNDRY_API_KEY:
        return False, "AZURE_FOUNDRY_API_KEY is not set (put it in your .env file)."
    if not config.FOUNDRY_DEPLOYMENT:
        return False, "AZURE_FOUNDRY_DEPLOYMENT is not set (e.g. gpt-4o-mini)."
    return True, "Azure AI Foundry is configured."


def status() -> dict:
    """Human-readable configuration state, used by the app status panel."""
    ok, message = is_configured()
    return {
        "configured": ok,
        "message": message,
        "endpoint": config.FOUNDRY_ENDPOINT or "(not set)",
        "deployment": config.FOUNDRY_DEPLOYMENT or "(not set)",
        "api_version": config.FOUNDRY_API_VERSION,
    }


def _get_client():
    global _client
    if _client is not None:
        return _client

    ok, message = is_configured()
    if not ok:
        raise FoundryNotConfigured(message)
    try:
        from openai import AzureOpenAI  # type: ignore
    except ImportError as exc:
        raise FoundryNotConfigured(
            "The openai package is missing. Run: pip install -r requirements.txt"
        ) from exc

    _client = AzureOpenAI(
        api_key=config.FOUNDRY_API_KEY,
        api_version=config.FOUNDRY_API_VERSION,
        azure_endpoint=config.FOUNDRY_ENDPOINT,
    )
    return _client


def complete(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """Low-level chat call. Any feature can use this for its own prompts."""
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=config.FOUNDRY_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # network, auth, quota, wrong deployment name...
        raise FoundryError("Azure AI Foundry request failed: " + str(exc)) from exc

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise FoundryError("Azure AI Foundry returned an empty response.")
    return strip_wrapping_quotes(text)


def generate_text(system_prompt: str, user_prompt: str) -> str:
    """The shared text helper named in the hearing flow's Foundry contract.

    ``features/hearing.py`` looks this name up first, so keep the signature
    ``(system_prompt, user_prompt) -> str`` stable. Raising is fine: the caller
    catches it and shows the message.
    """
    return complete(system_prompt, user_prompt)


def strip_wrapping_quotes(text: str) -> str:
    """Models sometimes wrap the answer in quotes; a voice would read them out."""
    text = text.strip()
    pairs = (
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("«", "»"),
    )
    for left, right in pairs:
        if len(text) > 1 and text.startswith(left) and text.endswith(right):
            return text[1:-1].strip()
    return text


# --- Task helpers ------------------------------------------------------------


def transform_text(text: str, mode_key: str) -> str:
    """Rewrite ``text`` according to one of the MODES. Meaning is preserved."""
    text = (text or "").strip()
    if not text:
        return ""

    mode = get_mode(mode_key)
    if mode.key == "exact":
        return text

    system = (
        "You are the rewriting engine of an assistive communication app. The "
        "person using it has a speech impairment and types what they want to "
        "say; your output is spoken out loud on their behalf. You are their "
        "voice, not their author.\n\n"
        "Task: " + mode.instruction + "\n\n" + _MEANING_GUARD
    )
    return complete(system, text, temperature=0.2, max_tokens=400)


def translate_text(
    text: str,
    target_language: str,
    *,
    tone_hint: Optional[str] = None,
) -> str:
    """Translate ``text`` into ``target_language`` (a name such as Romanian)."""
    text = (text or "").strip()
    if not text:
        return ""

    system = (
        "You are the translation engine of an assistive communication app. The "
        "translation is spoken out loud by a speech synthesizer on behalf of a "
        "person with a speech impairment.\n\n"
        "Task: Translate the input into " + target_language + ". Produce the "
        "natural way a native speaker would say this out loud in a real "
        "conversation - idiomatic, not word for word. Keep names, numbers and "
        "units unchanged.\n\n"
        + ("Keep this tone: " + tone_hint + ".\n\n" if tone_hint else "")
        + "Absolute rules:\n"
        "- Preserve the meaning exactly. Never add or drop information.\n"
        "- Keep the same point of view and the same level of politeness.\n"
        "- The text will be read by a speech synthesizer: plain speakable words, "
        "no markdown, no emoji, no quotation marks around the answer.\n"
        "- Reply with the " + target_language + " text only, nothing else."
    )
    return complete(system, text, temperature=0.2, max_tokens=600)
