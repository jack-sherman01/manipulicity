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
from typing import Union

import numpy as np
from openai import OpenAI
from PIL import Image


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a robotics perception expert specialising in estimating the physical \
properties of objects from visual information.

Given an image that shows an object grasped by a robot gripper, or hanging \
from a robotic end-effector, your task is to estimate the **mass** of that \
object in kilograms.

Reason step-by-step, considering:
1. The visible size and shape of the object.
2. The likely material (metal, plastic, wood, food, fabric, etc.) judged from \
   colour, texture, and context.
3. Typical real-world densities and dimensions for objects of that class.
4. Any scale references visible in the image (robot links, gripper fingers, \
   background objects).

After reasoning, output ONLY valid JSON with exactly these keys:
{
  "mass_kg": <float>,
  "mass_kg_range": [<float_lower>, <float_upper>],
  "material_guess": "<string>",
  "object_description": "<string>",
  "confidence": "<low|medium|high>",
  "reasoning": "<concise explanation>"
}

Do not output anything outside the JSON block.
"""

USER_PROMPT = (
    "Please estimate the mass of the object shown in this image. "
    "The object is being grasped or is hanging from a robot end-effector."
)


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
        OpenAI vision model to use.  Defaults to ``"gpt-4o-mini"``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An OpenAI API key is required. Pass it via the `api_key` "
                "argument or set the OPENAI_API_KEY environment variable."
            )
        self._client = OpenAI(api_key=resolved_key)
        self.model = model

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
        """Send the image to the VLM and return the raw text response."""
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
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _parse_response(text: str) -> dict:
        """
        Parse the VLM's JSON response, tolerating minor formatting issues
        (e.g. markdown code fences).
        """
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"VLM did not return valid JSON.\n"
                f"Raw response:\n{text}\n"
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
        "--model",
        default="gpt-4o-mini",
        help="OpenAI vision model to use (default: gpt-4o-mini).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    estimator = MassEstimator(api_key=args.api_key, model=args.model)

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
