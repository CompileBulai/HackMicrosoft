# 🌉 ContextBridge

An accessibility assistant that takes a real-world signal — a spoken
announcement, a scene, a voice — and turns it into whatever form the person in
front of it can actually use.

Built with Streamlit, Azure AI Speech and Azure AI Foundry.

---

## Status

| Flow | Owner | State |
|------|-------|-------|
| 👂 Hearing impairment | Person 2 | **Implemented** — see below |
| 🗣️ Vocal / speech impairment | Person 3 | **Implemented** — see below |
| 👁️ Visual impairment | teammate | placeholder only — drop your module in |
| Shared Foundry text helper (`services/foundry.py`) | Person 3 | **Implemented** |
| App shell (`app.py`) | integrator | **Implemented** |

---

## Setup

Requires **Python 3.11+** and, for the hearing flow, **ffmpeg** on `PATH`.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then fill in your Azure keys
.venv\Scripts\streamlit run app.py
```

The app starts even with no keys configured: the sidebar shows which Azure
services are missing, and anything that does not need them keeps working.

| Variable | Used by |
| --- | --- |
| `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | both flows — transcription and neural voices |
| `AZURE_FOUNDRY_ENDPOINT`, `AZURE_FOUNDRY_API_KEY`, `AZURE_FOUNDRY_DEPLOYMENT` | the shared text helper |

`.env` is git-ignored. `AZURE_OPENAI_*` names work as aliases.

Each flow also runs on its own, which is handy while developing:

```bash
streamlit run features\hearing.py
streamlit run features\vocal.py
```

---

## Layout

```
app.py               navigation + shared setup only, no feature logic
config.py            reads .env once, exposes typed constants
services/
  foundry.py         the only place we call a language model
  speech.py          the only place we call Azure Speech (STT + TTS)
features/
  hearing.py         👂  announcement -> captions and structured help
  vocal.py           🗣️  typed text -> natural speech, in any language
  visual.py          👁️  placeholder, waiting for its owner
```

`services/speech.py` has two halves that share one Azure Speech resource and
one `is_configured()`: `transcribe()` for the hearing flow, `synthesize()` plus
the voice catalogue for the vocal flow.

---

## Adding your feature

`app.py` imports each feature module lazily and never reaches inside it. The
whole contract is one function:

```python
# features/yours.py
def render() -> None:
    ...draw your Streamlit UI...
```

`main()`, `run()` and `app()` are accepted too. Two conventions keep the three
features from colliding:

* Prefix every `st.session_state` key with your feature name (`visual_`,
  `hearing_`, `vocal_`).
* Reuse `services/` instead of creating your own Azure clients:

```python
from services import foundry, speech

foundry.generate_text(system_prompt, user_prompt)   # -> str
speech.synthesize(text, voice_id)                   # -> mp3 bytes
speech.transcribe(audio_bytes, language="ro-RO")    # -> TranscriptionResult
```

Both service modules report readiness the same way:

```python
ok, message = speech.is_configured()      # (bool, str)
ok, message = foundry.is_configured()     # (bool, str)
```

Note the tuple — `if not speech.is_configured():` is always false, because a
non-empty tuple is truthy. Unpack it.

If your module raises on import or while rendering, `app.py` shows the traceback
on **your** page only — the other flows keep working.

---

## 👂 Hearing impairment flow

Record a spoken announcement, get it written down, and get exactly the kind of
help you asked for.

**Files:** [`services/speech.py`](services/speech.py) (STT half),
[`features/hearing.py`](features/hearing.py)

- Records through `st.audio_input`, with a file upload as a fallback for when
  a microphone is not available at demo time.
- Normalises the recording to 16 kHz mono 16-bit PCM WAV with ffmpeg. Browsers
  hand Streamlit whatever their MediaRecorder produced (usually WebM/Opus), and
  the Speech SDK will not read that without extra codecs installed.
- Transcribes with **continuous recognition**, not `recognize_once`, so a long
  announcement or one with a pause in the middle is captured in full instead of
  being cut off at the first silence.
- Produces per-phrase captions with `mm:ss` timestamps.
- Lets the user pick the spoken language (8 languages) and the language of the
  result. The result language follows the spoken language until the user picks
  one explicitly; after that their choice is kept.
- Lets the user tick any combination of seven kinds of help. The Foundry prompt
  is built from only the boxes that are ticked:

  | Option | Rendered as |
  |--------|-------------|
  | Show full captions / transcript | timestamped caption track (answered locally) |
  | Extract main message | Main message |
  | Highlight required action | What you have to do |
  | Highlight location | Where |
  | Highlight time / deadline | When |
  | Simplify the announcement | In simple words |
  | Reduce information overload | Only what matters (max 4 bullets) |

- Caches the Foundry answer, so ticking a different option does not re-record
  or re-transcribe anything.
- Offers the whole result as a downloadable `.txt`.

**The "never invent" guarantee.** The system prompt forbids adding any detail
not present in the transcript and requires the literal string `Not mentioned`
for anything absent. On top of that, the app treats a *missing* JSON key as
`Not mentioned` too — so a field the model silently drops still renders as
"Not mentioned in the announcement" rather than disappearing or being filled in
from somewhere else.

---

## 🗣️ Vocal / speech impairment flow

For people who cannot speak, or whose speech is hard for strangers to
understand. Type it, or type just the keywords, and the app says it for you.

**Files:** [`services/speech.py`](services/speech.py) (TTS half),
[`services/foundry.py`](services/foundry.py),
[`features/vocal.py`](features/vocal.py)

```
typed text
  -> Azure AI Foundry rewrite      (optional, meaning preserved)
  -> Azure AI Foundry translation  (optional, into the chosen language)
  -> Azure AI Speech synthesis     (neural voice for that language)
```

| Option | Effect |
| --- | --- |
| Speak exactly what I typed | No model call at all — your exact words. |
| Turn keywords into a full sentence | `water cold please` → *Could I have some cold water, please?* |
| Make it clearer | Understandable the first time it is heard. |
| Make it shorter | Trims filler, never information. |
| Make it more polite | Adds warmth; the request stays identical. |
| Make it more formal | Professional register for work or medical settings. |

**Speaking in another language.** Pick any of the 19 supported languages and
turn on *Translate my message into …*. The text is translated to sound like a
native speaker saying it out loud, then spoken by a neural voice belonging to
that locale — multilingual voices such as Ava, Vivienne and Seraphina where they
exist — so it sounds natural rather than like a faked accent. Leave the toggle
off if you already typed in that language.

**Safeguards**

- Every prompt forbids adding, removing or reinterpreting information; the model
  rewrites wording only.
- The original, the rewrite and the translation are all shown side by side, and
  the exact string that will be spoken is displayed before you press play.
- If Foundry fails, nothing is changed and you can still speak your own words.
- Quick phrases cover urgent needs in one tap; the last five messages can be
  replayed without regenerating them.
- Speaking speed, pitch and (where the voice supports it) speaking style are
  adjustable.

---

## Foundry contract

`services/foundry.py` is the only place the app talks to a language model.

```python
def generate_text(system_prompt: str, user_prompt: str) -> str: ...
def complete(system: str, user: str, *, temperature=0.2, max_tokens=800) -> str: ...
def transform_text(text: str, mode_key: str) -> str: ...
def translate_text(text: str, target_language: str) -> str: ...
```

`features/hearing.py` resolves the helper by name and adapts to its arity, so
`generate_text` must keep the signature `(system_prompt, user_prompt) -> str`.

---

## What has been verified

Checked by actually running the code through Streamlit's `AppTest` harness:

- All four pages (home, visual, hearing, vocal) render with **zero exceptions**,
  both with and without credentials.
- Hearing: audio conversion on a real WebM/Opus file → 16 kHz mono WAV; the
  result-language follow/pin behaviour; Foundry response parsing for clean JSON,
  fenced JSON, JSON wrapped in prose, missing keys (→ `Not mentioned`),
  malformed JSON and non-JSON replies; helper arity adaptation.
- Vocal: all 19 language pages; all 6 transformation modes; keywords → sentence
  → speech; translation into Romanian spoken by `ro-RO-AlinaNeural`; speed and
  pitch reaching the synthesizer; SSML escaping of `&` and `<`.
- Error paths surface readable messages instead of tracebacks: missing
  credentials, missing SDK, empty recording, undecodable audio, Foundry failure,
  synthesis cancellation.
- A feature module that raises is contained to its own page; the other flows
  keep working.

**Not yet verified:** the live Azure round-trip for either flow — no
subscription key has been configured yet. Everything up to and including the
call into the SDK is exercised; the network call itself is not.
