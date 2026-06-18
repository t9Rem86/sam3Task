"""
End-to-end pipeline: (YouTube URL | local file) -> annotated video.

SAM 3 (via 🤗 Transformers) performs Promptable Concept Segmentation on images,
so a video is processed frame by frame: each frame is segmented with every
concept phrase, then masks/boxes/labels are drawn and written to the output.
"""

import os
import logging
import cv2
from tqdm import tqdm

from .downloader import is_youtube_url, download_youtube_video
from .annotator import draw_detections, draw_legend
from .config import CONCEPT_GROUPS, DEFAULT_CONCEPTS

logger = logging.getLogger(__name__)


def _resolve_source(source: str) -> str:
    """Return a local video path, downloading from YouTube if needed."""
    if is_youtube_url(source):
        logger.info("Source is a YouTube URL — downloading...")
        return download_youtube_video(source)
    if not os.path.exists(source):
        raise FileNotFoundError(f"Video file not found: {source}")
    return source


def _category_colors() -> dict[str, tuple]:
    return {g["label"]: g["color"] for g in CONCEPT_GROUPS.values()}


def _counts(detections: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in detections:
        counts[d["category"]] = counts.get(d["category"], 0) + 1
    return counts


def run(
    source: str,
    output_path: str = "output.mp4",
    concepts: list[str] | None = None,
    device: str = "auto",
    score_thresh: float = 0.5,
    mask_thresh: float = 0.5,
    max_frames: int | None = None,
    frame_stride: int = 1,
) -> str:
    """Process a video and write an annotated copy to output_path."""
    concepts = concepts or DEFAULT_CONCEPTS
    for c in concepts:
        if c not in CONCEPT_GROUPS:
            raise ValueError(f"Unknown concept '{c}'. Valid: {list(CONCEPT_GROUPS)}")

    video_path = _resolve_source(source)
    logger.info(f"Processing: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / max(1, frame_stride)
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
    colors = _category_colors()

    # Import here so that --help etc. work without heavy ML deps installed.
    from .sam3_segmenter import Sam3ConceptSegmenter
    seg = Sam3ConceptSegmenter(
        concepts, score_thresh=score_thresh, mask_thresh=mask_thresh, device=device
    )

    n_expected = max_frames or (total // frame_stride if total else None)
    pbar = tqdm(total=n_expected, desc="Segmenting", unit="frame")
    frame_idx = processed = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                dets = seg.segment_frame(frame_rgb)
                annotated = draw_detections(frame_bgr, dets)
                annotated = draw_legend(annotated, _counts(dets), colors)
                writer.write(annotated)
                processed += 1
                pbar.update(1)
                if max_frames and processed >= max_frames:
                    break
            frame_idx += 1
    finally:
        pbar.close()
        cap.release()
        writer.release()

    logger.info(f"Done. Processed {processed} frames -> {output_path}")
    return output_path
