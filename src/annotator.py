"""Draw SAM 3 masks, bounding boxes and labels onto frames (OpenCV / BGR)."""

import cv2
import numpy as np

_MASK_ALPHA = 0.45
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Return a copy of frame_bgr with all detections drawn on it."""
    out = frame_bgr.copy()
    overlay = frame_bgr.copy()

    # 1. semi-transparent mask fills
    for det in detections:
        mask = det.get("mask")
        if mask is None:
            continue
        if mask.shape[:2] != out.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (out.shape[1], out.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay[mask] = det["color"]
    cv2.addWeighted(overlay, _MASK_ALPHA, out, 1 - _MASK_ALPHA, 0, out)

    # 2. boxes + labels on top
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = det["color"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = det["category"]
        if det.get("track_id", -1) >= 0:
            label += f" #{det['track_id']}"
        label += f" {det['score']:.2f}"

        (tw, th), base = cv2.getTextSize(label, _FONT, 0.5, 1)
        ly = max(y1, th + 4)
        cv2.rectangle(out, (x1, ly - th - base - 2), (x1 + tw + 4, ly), color, -1)
        cv2.putText(out, label, (x1 + 2, ly - base),
                    _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def draw_legend(frame_bgr: np.ndarray, counts: dict[str, int],
                colors: dict[str, tuple]) -> np.ndarray:
    """Draw a small per-category counter panel in the top-left corner."""
    out = frame_bgr
    x, y = 10, 24
    for category, count in counts.items():
        color = colors.get(category, (255, 255, 255))
        cv2.rectangle(out, (x, y - 14), (x + 16, y + 2), color, -1)
        cv2.putText(out, f"{category}: {count}", (x + 24, y),
                    _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"{category}: {count}", (x + 24, y),
                    _FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 26
    return out
