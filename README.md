# Manipulicity — VLM-Based Physical Mass Estimation

This branch explores using **Vision-Language Models (VLMs)** to infer the
physical parameters of objects (starting with **mass**) from visual input.
The estimated mass will later feed into an impedance controller for compliant
robot interaction.

---

## What it does

Given a single RGB image of an object that is **grasped** or **hanging** from
a robot end-effector, `mass_estimator.py` queries an OpenAI vision model
(GPT-4o / GPT-4o-mini) and returns:

| Field | Description |
|---|---|
| `mass_kg` | Best-estimate mass (float, kg) |
| `mass_kg_range` | Plausible `[lower, upper]` range (kg) |
| `material_guess` | Inferred material (e.g. "metal", "plastic") |
| `object_description` | Brief description of the object |
| `confidence` | `"low"` / `"medium"` / `"high"` |
| `reasoning` | Chain-of-thought explanation |

---

## Installation

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

---

## Usage

### Command line

```bash
python mass_estimator.py --image path/to/image.jpg
# optional flags:
#   --api_key  sk-...          (or set OPENAI_API_KEY env var)
#   --model    gpt-4o          (default: gpt-4o-mini)
```

### Python API

```python
from mass_estimator import MassEstimator

estimator = MassEstimator(api_key="sk-...")   # or rely on env var

# Accept file path, PIL.Image, or numpy ndarray
result = estimator.estimate("gripper_holding_mug.jpg")

print(result["mass_kg"])        # e.g. 0.35
print(result["confidence"])     # e.g. "medium"
print(result["reasoning"])
```

---

## Project structure

```
manipulicity/
├── mass_estimator.py   # Core VLM mass-estimation module + CLI
├── requirements.txt    # Minimal Python dependencies
└── README.md           # This file
```

---

## Notes

- GPT-4o-mini provides a good cost/accuracy balance for most objects.
  Switch to `--model gpt-4o` for higher accuracy on ambiguous scenes.
- The estimated mass is a *semantic* estimate based on visual appearance and
  world knowledge; it is not a measurement.  For impedance control the value
  is used as a soft prior that can be refined online.
