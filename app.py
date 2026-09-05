"""ContextBridge - one app, three accessibility bridges.

app.py owns navigation and shared setup only. All feature logic lives in its own
module under features/, and all Azure access lives under services/. If you are
adding behaviour, it almost certainly does not belong in this file.
"""

from __future__ import annotations

import importlib
import traceback
from dataclasses import dataclass
from types import ModuleType
from typing import Optional

import streamlit as st

import config
from services import foundry, speech

# Function names app.py will accept as a feature's entry point, in order of
# preference. Feature owners can use whichever they already wrote.
ENTRY_POINTS = ("render", "main", "run", "app")


@dataclass(frozen=True)
class Category:
    key: str
    title: str
    icon: str
    module: str
    tagline: str

    @property
    def nav_label(self) -> str:
        return self.icon + "  " + self.title


CATEGORIES: tuple[Category, ...] = (
    Category(
        key="visual",
        title="Visual impairment",
        icon="👁️",
        module="features.visual",
        tagline="See with your ears - images, documents and scenes described out loud.",
    ),
    Category(
        key="hearing",
        title="Hearing impairment",
        icon="👂",
        module="features.hearing",
        tagline="Hear with your eyes - speech turned into live captions and transcripts.",
    ),
    Category(
        key="vocal",
        title="Vocal / speech impairment",
        icon="🗣️",
        module="features.vocal",
        tagline="Speak with your keyboard - typed words and keywords spoken in a natural voice.",
    ),
)

CATEGORIES_BY_KEY = {category.key: category for category in CATEGORIES}
HOME = "home"


# --- Feature loading ---------------------------------------------------------


def load_feature(category: Category) -> tuple[Optional[ModuleType], Optional[BaseException]]:
    """Import a feature module, returning the error instead of raising it.

    A half-finished module from one feature owner must never take the whole app
    down for the other two.
    """
    try:
        return importlib.import_module(category.module), None
    except BaseException as exc:  # noqa: BLE001 - we deliberately show anything
        return None, exc


def render_feature(category: Category) -> None:
    module, error = load_feature(category)

    if error is not None:
        st.header(category.icon + " " + category.title)
        st.error(
            "This feature could not be loaded, so the rest of the app kept running. "
            "The module `" + category.module + "` raised " + type(error).__name__ + ".",
            icon="🧩",
        )
        with st.expander("Details for the feature owner"):
            st.code("".join(traceback.format_exception(error)))
        return

    entry = next(
        (getattr(module, name) for name in ENTRY_POINTS if callable(getattr(module, name, None))),
        None,
    )
    if entry is None:
        st.header(category.icon + " " + category.title)
        st.error(
            "`" + category.module + "` has no entry point. Add a `render()` function "
            "that draws the feature (or name it " + ", ".join(ENTRY_POINTS[1:]) + ").",
            icon="🧩",
        )
        return

    try:
        entry()
    except Exception as exc:  # noqa: BLE001 - keep the shell alive
        st.error(
            "This feature stopped with " + type(exc).__name__ + ": " + str(exc),
            icon="⚠️",
        )
        with st.expander("Details for the feature owner"):
            st.code(traceback.format_exc())


# --- Shell -------------------------------------------------------------------


def render_sidebar() -> str:
    with st.sidebar:
        st.title("🌉 " + config.APP_NAME)
        st.caption(config.APP_TAGLINE)

        st.markdown("### Choose what you need")
        options = [HOME] + [category.key for category in CATEGORIES]

        def label(key: str) -> str:
            if key == HOME:
                return "🏠  Home"
            return CATEGORIES_BY_KEY[key].nav_label

        selection = st.radio(
            "Category",
            options=options,
            format_func=label,
            key="nav",
            label_visibility="collapsed",
        )

        st.divider()
        render_status()
        return selection


def render_status() -> None:
    foundry_status = foundry.status()
    speech_status = speech.status()
    ready = foundry_status["configured"] and speech_status["configured"]

    with st.expander("Azure services" + ("" if ready else "  ⚠️"), expanded=not ready):
        st.markdown(
            ("✅" if foundry_status["configured"] else "❌")
            + " **AI Foundry** - text understanding and rewriting"
        )
        if foundry_status["configured"]:
            st.caption("Deployment: " + foundry_status["deployment"])

        st.markdown(
            ("✅" if speech_status["configured"] else "❌")
            + " **AI Speech** - text-to-speech and speech-to-text"
        )
        if speech_status["configured"]:
            st.caption("Region: " + speech_status["region"])

        if not ready:
            st.caption(
                "Copy `.env.example` to `.env` and fill in your Azure keys, then "
                "restart the app. Features that do not need the missing service "
                "keep working."
            )


def render_home() -> None:
    st.title("🌉 " + config.APP_NAME)
    st.subheader(config.APP_TAGLINE)
    st.write(
        "Communication breaks down in different places for different people. "
        "ContextBridge fixes the broken step and leaves the rest of the "
        "conversation alone - it never changes what you mean."
    )

    st.divider()
    columns = st.columns(3)
    for column, category in zip(columns, CATEGORIES):
        with column:
            st.markdown("## " + category.icon)
            st.markdown("#### " + category.title)
            st.write(category.tagline)
            if st.button("Open", key="home_open_" + category.key, use_container_width=True):
                st.session_state["nav"] = category.key
                st.rerun()

    st.divider()
    st.caption(
        "Built on Azure AI Foundry and Azure AI Speech. "
        "Pick a category on the left to begin."
    )


def main() -> None:
    st.set_page_config(
        page_title=config.APP_NAME,
        page_icon="🌉",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    selection = render_sidebar()
    if selection == HOME:
        render_home()
    else:
        render_feature(CATEGORIES_BY_KEY[selection])


if __name__ == "__main__":
    main()
