from __future__ import annotations

import base64
import os
from functools import lru_cache

from openai import OpenAI


class FoundryVisionError(RuntimeError):
    """Raised when the shared Foundry vision service cannot complete a request."""


def _base_url() -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    if not endpoint:
        raise FoundryVisionError("AZURE_OPENAI_ENDPOINT is not configured.")

    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/openai/v1"):
        return f"{endpoint}/"
    return f"{endpoint}/openai/v1/"


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    if not api_key:
        raise FoundryVisionError("AZURE_OPENAI_API_KEY is not configured.")

    return OpenAI(
        base_url=_base_url(),
        api_key=api_key,
    )


def analyze_image(
    *,
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    deployment: str | None = None,
    max_output_tokens: int = 900,
) -> str:
    """
    Analyze an image with a vision-capable Azure OpenAI deployment in Microsoft Foundry.

    Required environment variables:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_VISION_DEPLOYMENT

    AZURE_OPENAI_ENDPOINT can be the resource root, for example:
      https://YOUR-RESOURCE-NAME.openai.azure.com
    """
    if not image_bytes:
        raise FoundryVisionError("No image data was supplied.")
    if not prompt.strip():
        raise FoundryVisionError("The vision prompt is empty.")

    deployment_name = (
        deployment or os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT", "")
    ).strip()
    if not deployment_name:
        raise FoundryVisionError(
            "AZURE_OPENAI_VISION_DEPLOYMENT is not configured."
        )

    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"

    try:
        response = _client().responses.create(
            model=deployment_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": data_url,
                            "detail": "high",
                        },
                    ],
                }
            ],
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        raise FoundryVisionError(str(exc)) from exc

    output_text = (response.output_text or "").strip()
    if not output_text:
        raise FoundryVisionError("The vision model returned an empty response.")

    return output_text
