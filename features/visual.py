from __future__ import annotations

from typing import Callable

import streamlit as st

from services.foundry import FoundryVisionError, analyze_image


TASKS = [
    "Read signs and text",
    "Describe surroundings",
    "Find entrances, exits, elevators",
    "Spot obvious obstacles",
    "Find room numbers, gate numbers, times",
    "Tell me what to do next",
    "Find something specific",
    "Reduce information overload",
]

TASK_INSTRUCTIONS: dict[str, Callable[[str | None], str]] = {
    "Read signs and text": lambda _: (
        "Read visible signs, labels, notices, and other text that is clearly legible. "
        "Quote only text you can actually read."
    ),
    "Describe surroundings": lambda _: (
        "Describe the visible surroundings, layout, and notable objects that would help "
        "a person understand the scene."
    ),
    "Find entrances, exits, elevators": lambda _: (
        "Locate visible entrances, exits, elevators, and related wayfinding signs. "
        "Describe their approximate position in the image."
    ),
    "Spot obvious obstacles": lambda _: (
        "Point out obvious visible obstacles such as steps, poles, barriers, clutter, "
        "open doors, or objects blocking a path. Do not claim that a route is safe."
    ),
    "Find room numbers, gate numbers, times": lambda _: (
        "Look specifically for clearly legible room numbers, gate numbers, platform numbers, "
        "opening/closing times, departure times, or other relevant numbers and times."
    ),
    "Tell me what to do next": lambda _: (
        "Suggest a practical next action based only on what is visibly supported by the image. "
        "Do not provide safety-critical navigation or claim a route is safe."
    ),
    "Find something specific": lambda target: (
        f"Look for this specific target: {target.strip()}." if target and target.strip()
        else "Look for the user's specific target, but do not guess if it is not visible."
    ),
    "Reduce information overload": lambda _: (
        "Prioritize only the most relevant information for the selected tasks and omit "
        "nonessential visual detail."
    ),
}

STYLE_INSTRUCTIONS = {
    "concise": "Keep the answer brief and prioritize the most important information first.",
    "step-by-step": (
        "Present the answer as a short ordered sequence of actionable steps where appropriate."
    ),
    "detailed": (
        "Give a thorough answer while staying grounded in visible evidence and clearly separating "
        "confirmed details from uncertain ones."
    ),
}


def build_visual_prompt(
    selected_tasks: list[str],
    response_style: str,
    specific_target: str | None = None,
) -> str:
    """Build a task-specific prompt without adding unselected assistance tasks."""
    task_lines = [
        f"- {TASK_INSTRUCTIONS[task](specific_target)}"
        for task in selected_tasks
        if task in TASK_INSTRUCTIONS
    ]

    if not task_lines:
        raise ValueError("At least one assistance task must be selected.")

    style_instruction = STYLE_INSTRUCTIONS.get(
        response_style,
        STYLE_INSTRUCTIONS["concise"],
    )

    return f"""
You are the visual assistance component of ContextBridge.

Follow these global rules:
- Never identify, name, recognize, or infer the identity of any person in the image.
- If people are visible, refer to them only generically, such as "a person" or "two people".
- Never invent text. If text is blurry, cut off, too small, obscured, or otherwise unreadable,
  explicitly say that it is unreadable or uncertain.
- Clearly communicate uncertainty using phrases such as "appears to", "possibly", or
  "I can't confirm from this image" when evidence is incomplete.
- Do not claim that any route, crossing, doorway, staircase, platform edge, or environment is safe.
- Do not provide safety-critical navigation. You may describe visible landmarks and suggest
  non-safety-critical next steps, but make clear that the user should verify surroundings directly
  or with an appropriate trusted source when safety matters.
- Base the answer only on visible evidence in the supplied image.

Perform only these selected assistance tasks:
{chr(10).join(task_lines)}

Response style:
- {style_instruction}
""".strip()


def render() -> None:
    """Render the ContextBridge Visual Impairment feature."""
    st.header("Visual assistance")
    st.caption(
        "Upload a photo or take one with your camera, then choose exactly what kind of help you want."
    )

    source = st.radio(
        "Image source",
        options=("Upload image", "Use camera"),
        horizontal=True,
    )

    image_file = None
    if source == "Upload image":
        image_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
        )
    else:
        image_file = st.camera_input("Take a photo")

    selected_tasks = st.multiselect(
        "What help do you want?",
        options=TASKS,
        placeholder="Choose one or more assistance tasks",
    )

    specific_target = None
    if "Find something specific" in selected_tasks:
        specific_target = st.text_input(
            "What should I look for?",
            placeholder="For example: the blue check-in desk, restroom sign, Gate 14",
        )

    response_style = st.radio(
        "Response style",
        options=("concise", "step-by-step", "detailed"),
        horizontal=True,
        index=0,
    )

    st.info(
        "This feature describes what is visible in the image. It does not identify people, "
        "guess unreadable text, or certify that a route or environment is safe."
    )

    can_analyze = image_file is not None and bool(selected_tasks)
    if "Find something specific" in selected_tasks and not (specific_target or "").strip():
        can_analyze = False

    if st.button("Analyze image", type="primary", disabled=not can_analyze):
        try:
            image_bytes = image_file.getvalue()
            mime_type = getattr(image_file, "type", None) or "image/jpeg"
            prompt = build_visual_prompt(
                selected_tasks=selected_tasks,
                response_style=response_style,
                specific_target=specific_target,
            )

            with st.spinner("Analyzing image..."):
                result = analyze_image(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    prompt=prompt,
                )

            st.subheader("Visual assistance")
            st.write(result)

        except ValueError as exc:
            st.warning(str(exc))
        except FoundryVisionError as exc:
            st.error(
                "The vision service could not analyze this image. "
                f"Technical detail: {exc}"
            )
        except Exception:
            st.error(
                "Something unexpected went wrong while analyzing the image. "
                "Please try again with another image."
            )


if __name__ == "__main__":
    render()
