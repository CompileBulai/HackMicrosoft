# ContextBridge

An accessibility assistant that takes a real-world signal — a spoken
announcement, a scene, a voice — and turns it into whatever form the person in
front of it can actually use.

Built with Streamlit, Azure AI Speech and Azure AI Foundry.

---

## Status

| Flow | Owner | State |
|------|-------|-------|
| Hearing impairment | Person 2 | **Implemented** — see below |
| Visual impairment | teammate | not started in this repo |
| Vocal impairment | teammate | not started in this repo |
| Shared Foundry text helper (`services/foundry.py`) | teammate | **not present yet** — see [contract](#foundry-contract) |
| App shell (`app.py`) | integrator | not started in this repo |

---

## Hearing impairment flow — done

Record a spoken announcement, get it written down, and get exactly the kind of
help you asked for.

**Files**

- [`services/speech.py`](services/speech.py) — all Azure Speech SDK code
- [`features/hearing.py`](features/hearing.py) — the Streamlit flow

**What it does**

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

**The "never invent" guarantee**

The system prompt forbids adding any detail not present in the transcript and
requires the literal string `Not mentioned` for anything absent. On top of that,
the app treats a *missing* JSON key as `Not mentioned` too — so a field the
model silently drops still renders as "Not mentioned in the announcement"
rather than disappearing or being filled in from somewhere else.

---

## Foundry contract

`features/hearing.py` does not talk to Foundry directly. It expects
`services/foundry.py` to expose:

```python
def generate_text(system_prompt: str, user_prompt: str) -> str: ...
```

Details for whoever writes it:

- The helper must return the model's text reply. The hearing flow asks for a
  single JSON object and parses it, tolerating code fences and surrounding
  prose, so the helper does not need to clean anything up.
- These alternative names are also accepted: `run_text_task`, `ask_foundry`,
  `complete`, `chat`, `ask`, `run`.
- A one-argument helper works too — the two prompts are joined before the call.

Until the file exists, the hearing flow still runs and shows the transcript and
captions, with a message naming exactly what is missing.

---

## Setup

Requires **Python 3.12** and **ffmpeg** on `PATH`.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then fill in your Azure Speech key
```

`.env`:

```
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=westeurope
```

Run the hearing flow on its own:

```bash
.venv\Scripts\streamlit run features\hearing.py
```

Or mount it in the shared app with `features.hearing.render()`.

Pinned in `requirements.txt`: `streamlit>=1.41` (`st.audio_input` was added
there), `azure-cognitiveservices-speech>=1.40`, `python-dotenv>=1.0`.
Verified against streamlit 1.63.0 and azure-cognitiveservices-speech 1.51.2.

---

## What has been verified

Checked by actually running the code:

- Both modules import and render with **zero exceptions** through Streamlit's
  `AppTest` harness.
- Audio conversion on a real WebM/Opus file → 16 kHz mono 16-bit WAV, correct
  duration.
- Result-language follow/pin behaviour across five language switches.
- Foundry response parsing: clean JSON, fenced JSON, JSON wrapped in prose,
  missing keys (→ `Not mentioned`), malformed JSON and non-JSON replies
  (→ the raw answer is shown rather than an empty screen).
- Foundry helper arity adaptation, and the message shown while
  `services/foundry.py` is absent.
- Error paths surface readable messages instead of tracebacks: missing
  credentials, missing SDK, empty recording, undecodable audio.

**Not yet verified:** transcription against the live Azure Speech service —
no subscription key has been configured yet. Everything up to and including the
call into the SDK is exercised; the network round-trip is not.
