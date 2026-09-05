"""Visual impairment feature - PLACEHOLDER.

This file exists only so the app runs end to end before every feature has
landed. Replace the whole file with the real implementation; nothing here is
worth keeping.

Integration contract (the only thing app.py needs):

    def render() -> None:
        ...draw your Streamlit UI here...

app.py also accepts ``main()``, ``run()`` or ``app()`` if that is what you
already wrote. Shared helpers you can reuse instead of rebuilding:

    from services import foundry     # foundry.complete(system, user) -> str
    from services import speech      # speech.synthesize(text, voice_id) -> mp3 bytes

Prefix any st.session_state keys with "visual_" so the three features do not
collide.
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("👁️ Visual impairment")
    st.caption("Describe the world around you: images, documents and scenes read out loud.")
    st.info(
        "This feature is still being built by its owner. "
        "Replace `features/visual.py` with the real implementation - app.py will "
        "pick it up automatically, no changes needed here.",
        icon="🚧",
    )
