"""
Concept definitions for SAM 3 Promptable Concept Segmentation (PCS).

Each concept group maps a category name to:
  - prompts : list of short text phrases fed to SAM 3 as concept prompts
  - color   : BGR colour used to draw masks/boxes (OpenCV uses BGR)

SAM 3 segments *all instances* of an open-vocabulary concept given a short
text phrase, so we simply describe what we want in plain English.
"""

# BGR colours (OpenCV)
_GREEN  = (50, 205, 50)
_ORANGE = (0, 140, 255)
_YELLOW = (0, 230, 230)

CONCEPT_GROUPS: dict[str, dict] = {
    "workers": {
        "prompts": [
            "construction worker",
            "person wearing a hard hat",
            "worker in a safety vest",
        ],
        "color": _GREEN,
        "label": "Worker",
    },
    "machinery": {
        "prompts": [
            "construction machinery",
            "excavator",
            "bulldozer",
            "wheel loader",
            "dump truck",
            "crane",
        ],
        "color": _ORANGE,
        "label": "Machinery",
    },
    "license_plate": {
        "prompts": [
            "vehicle license plate",
            "license plate",
        ],
        "color": _YELLOW,
        "label": "License Plate",
    },
}

# Default concept groups to run when the user does not specify any.
DEFAULT_CONCEPTS = ["workers", "machinery", "license_plate"]

# A reasonable, freely-usable construction-site demo video on YouTube.
# The user can override this with any file path or URL.
DEMO_YOUTUBE_URL = "https://www.youtube.com/watch?v=Ww4FX6h8aLk"
