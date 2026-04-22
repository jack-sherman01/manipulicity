"""
mass_estimator.py
-----------------
VLM-based physical mass estimation for objects grasped or hanging from a robot.

Usage (CLI):
    python mass_estimator.py --image path/to/image.jpg [--api_key sk-...] [--model gpt-4o-mini]

Usage (Python API):
    from mass_estimator import MassEstimator
    estimator = MassEstimator(api_key="sk-...")
    result = estimator.estimate(image)   # image: file path, PIL.Image, or np.ndarray
    print(result["mass_kg"], result["confidence"], result["reasoning"])
"""

import argparse
import base64
import io
import json
import os
import re
from typing import Optional, Union

import numpy as np
from openai import OpenAI
from PIL import Image

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a robotics perception expert specialising in estimating the physical \
properties of objects from visual information.

Given an image that shows an object grasped by a robot gripper, or hanging \
from a robotic end-effector, your task is to estimate the **mass** of that \
object in kilograms.

## Step 1 — Chain-of-Thought Reasoning (write this out freely)

Think through each of the following considerations explicitly before committing
to a number.  Write your reasoning in plain prose; do not skip any step:

1. **Object identity & description** – What is the object?  Describe its shape,
   size relative to the gripper/end-effector, and any distinguishing features.
2. **Material inference** – What material(s) is it likely made of (metal, plastic,
   wood, food, fabric, glass, etc.)?  Justify from colour, texture, sheen, and
   context clues visible in the image.
3. **Dimensional estimation** – Estimate the object's approximate dimensions
   (length × width × height or diameter × height) using the robot hardware as a
   scale reference.  State your scale assumptions explicitly.
4. **Mass calculation** – Using your estimated volume and a plausible density
   for the inferred material, compute an approximate mass.  Show the arithmetic.
5. **Sanity check** – Does the result feel physically reasonable for an object
   of that type?  Adjust and explain if needed.
6. **Uncertainty** – Note what makes this estimate uncertain and give a lower /
   upper bound.

## Step 2 — Structured Output

After your reasoning, output a JSON block (fenced with ```json ... ```) with
exactly these keys and no extra text outside the fences:

```json
{{
  "mass_kg": <float>,
  "mass_kg_range": [<float_lower>, <float_upper>],
  "material_guess": "<string>",
  "object_description": "<string>",
  "confidence": "<low|medium|high>",
  "reasoning": "<one-sentence summary of your reasoning>"
}}
```
"""

USER_PROMPT = (
    "Please estimate the mass of the object shown in this image. "
    "The object is being grasped or is hanging from a robot end-effector."
)

# Default local model served via Ollama (most capable open vision model as of 2025).
# Alternatives: "llama3.2-vision:90b" (largest), "qwen2.5vl:72b" (comparable).
_DEFAULT_LOCAL_MODEL = "llama3.2-vision:11b"


# ---------------------------------------------------------------------------
# Helper: image → base64 PNG
# ---------------------------------------------------------------------------

def _to_base64_png(image_input: Union[str, os.PathLike, np.ndarray, Image.Image]) -> str:
    """Convert any supported image format to a base64-encoded PNG string."""
    if isinstance(image_input, (str, bytes, os.PathLike)):
        with open(image_input, "rb") as f:
            raw = f.read()
        # Re-encode as PNG to normalise format
        buf = io.BytesIO(raw)
        pil_img = Image.open(buf).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, mode="L").convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            mode = "RGBA" if arr.shape[2] == 4 else "RGB"
            pil_img = Image.fromarray(arr, mode=mode).convert("RGB")
        else:
            raise ValueError(f"Unsupported numpy array shape: {arr.shape}")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(
            f"Expected file path, np.ndarray, or PIL.Image; got {type(image_input)}"
        )

    out = io.BytesIO()
    pil_img.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

class MassEstimator:
    """
    Estimate the mass of an object from a single RGB image using a VLM.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when not supplied.
    model : str
        Model to use.  Prefix with ``ollama/`` to route to a local server
        (e.g. ``"ollama/llama3.2-vision:11b"``).  Defaults to ``"gpt-4o-mini"``.
    gemini_api_key : str, optional
        Google API key for Gemini models.  Falls back to the
        ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` environment variables.
    local_base_url : str, optional
        Base URL of a local OpenAI-compatible VLM server (e.g. Ollama at
        ``http://localhost:11434/v1``).  Falls back to the
        ``LOCAL_VLM_BASE_URL`` environment variable, then
        ``http://localhost:11434/v1``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        gemini_api_key: Optional[str] = None,
        local_base_url: Optional[str] = None,
    ) -> None:
        self.model = model

        _model_lower = model.lower()
        if _model_lower.startswith("gemini"):
            self._provider = "gemini"
        elif _model_lower.startswith("ollama/") or local_base_url or os.environ.get("LOCAL_VLM_BASE_URL"):
            self._provider = "local"
        else:
            self._provider = "openai"

        if self._provider == "openai":
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "An OpenAI API key is required. Pass it via the `api_key` "
                    "argument or set the OPENAI_API_KEY environment variable."
                )
            self._client = OpenAI(api_key=resolved_key)
        else:
            if not _GEMINI_AVAILABLE:
                raise ImportError(
                    "google-generativeai is required for Gemini models. "
                    "Install it with: pip install google-generativeai"
                )
            resolved_key = (
                gemini_api_key
                or api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            if not resolved_key:
                raise ValueError(
                    "A Google API key is required for Gemini models. Pass it via "
                    "the `gemini_api_key` argument or set the GEMINI_API_KEY "
                    "environment variable."
                )
            genai.configure(api_key=resolved_key)
        else:  # local
            resolved_url = (
                local_base_url
                or os.environ.get("LOCAL_VLM_BASE_URL")
                or "http://localhost:11434/v1"
            )
            # Strip provider prefix (e.g. "ollama/llama3.2-vision:90b" → "llama3.2-vision:90b")
            self._local_model = model.split("/", 1)[1] if "/" in model else _DEFAULT_LOCAL_MODEL
            # Ollama requires a non-empty api_key string; any value works
            self._client = OpenAI(api_key="ollama", base_url=resolved_url)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def estimate(
        self,
        image: Union[str, os.PathLike, np.ndarray, Image.Image],
    ) -> dict:
        """
        Estimate the mass of the object visible in *image*.

        Parameters
        ----------
        image :
            The input image as a file path, ``np.ndarray`` (H×W×C, uint8 or
            float in [0,1]), or ``PIL.Image.Image``.

        Returns
        -------
        dict with keys:
            mass_kg          – best-estimate mass in kilograms (float)
            mass_kg_range    – [lower, upper] plausible range (list[float])
            material_guess   – inferred material (str)
            object_description – brief description (str)
            confidence       – "low", "medium", or "high" (str)
            reasoning        – chain-of-thought explanation (str)
        """
        image_b64 = _to_base64_png(image)
        raw_json = self._query_vlm(image_b64)
        return self._parse_response(raw_json)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _query_vlm(self, image_b64: str) -> str:
        """Route the image to the appropriate VLM and return the raw text response."""
        if self._provider == "gemini":
            return self._query_gemini(image_b64)
        if self._provider == "local":
            return self._query_local(image_b64)
        return self._query_openai(image_b64)

    def _query_openai(self, image_b64: str) -> str:
        """Send the image to OpenAI and return the raw text response."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def _query_local(self, image_b64: str) -> str:
        """Send the image to a local OpenAI-compatible VLM server (e.g. Ollama)."""
        response = self._client.chat.completions.create(
            model=self._local_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def _query_gemini(self, image_b64: str) -> str:
        """Send the image to Google Gemini and return the raw text response."""
        model_obj = genai.GenerativeModel(
            self.model,
            system_instruction=SYSTEM_PROMPT,
        )
        image_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(image_bytes))
        response = model_obj.generate_content(
            [USER_PROMPT, pil_img],
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )
        return response.text.strip()

    @staticmethod
    def _parse_response(text: str) -> dict:
        """
        Parse the VLM's response, which may contain free-form CoT reasoning
        followed by a JSON block (fenced or bare).  Extracts the JSON object
        and validates required keys.
        """
        # 1. Try to extract a fenced ```json ... ``` block first
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        candidate = fenced.group(1).strip() if fenced else text

        # 2. If no fence found, fall back to the last {...} object in the text
        if not fenced:
            brace_match = list(re.finditer(r"\{[\s\S]*\}", candidate))
            if brace_match:
                candidate = brace_match[-1].group(0)

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"VLM did not return valid JSON.\n"
                f"Extracted candidate:\n{candidate}\n"
                f"JSON error: {exc}"
            ) from exc

        required_keys = {
            "mass_kg",
            "mass_kg_range",
            "material_guess",
            "object_description",
            "confidence",
            "reasoning",
        }
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(
                f"VLM response is missing required keys: {missing}\n"
                f"Raw response:\n{text}"
            )

        # Coerce numeric types for safety
        data["mass_kg"] = float(data["mass_kg"])
        data["mass_kg_range"] = [float(v) for v in data["mass_kg_range"]]
        return data


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the mass of an object grasped or hanging from a robot "
            "end-effector using a Vision-Language Model."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image (JPEG / PNG / BMP …).",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--local_base_url",
        default=None,
        help=(
            "Base URL of a local OpenAI-compatible VLM server, e.g. "
            "http://localhost:11434/v1 for Ollama (falls back to "
            "LOCAL_VLM_BASE_URL env var, then http://localhost:11434/v1). "
            "Prefix the model name with 'ollama/' to auto-select this provider."
        ),
    )
    parser.add_argument(
        "--gemini_api_key",
        default=None,
        help=(
            "Google API key for Gemini models "
            "(falls back to GEMINI_API_KEY or GOOGLE_API_KEY env var)."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help=(
            "VLM to use: an OpenAI model (e.g. gpt-4o-mini, gpt-4o), a "
            "Gemini model (e.g. gemini-1.5-flash, gemini-1.5-pro), or a "
            "local model prefixed with 'ollama/' "
            "(e.g. ollama/llama3.2-vision:11b [recommended], "
            "ollama/llama3.2-vision:90b, ollama/qwen2.5vl:72b). "
            "Provider is auto-detected from the model name. "
            "Default: gpt-4o-mini."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    estimator = MassEstimator(
        api_key=args.api_key,
        model=args.model,
        gemini_api_key=args.gemini_api_key,
        local_base_url=args.local_base_url,
    )

    print(f"[MassEstimator] Querying {args.model} with image: {args.image}")
    result = estimator.estimate(args.image)

    print("\n===== Mass Estimation Result =====")
    print(f"  Object       : {result['object_description']}")
    print(f"  Material     : {result['material_guess']}")
    print(f"  Mass         : {result['mass_kg']:.4f} kg")
    print(
        f"  Range        : [{result['mass_kg_range'][0]:.4f}, "
        f"{result['mass_kg_range'][1]:.4f}] kg"
    )
    print(f"  Confidence   : {result['confidence']}")
    print(f"  Reasoning    : {result['reasoning']}")
    print("==================================\n")


if __name__ == "__main__":
    main()
