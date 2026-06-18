#!/usr/bin/env python3
"""
Construction site detection agent (SAM 3 / Meta).

Detects and highlights WORKERS, MACHINERY and LICENSE PLATES in a video,
given either a local file or a YouTube URL.

Examples
--------
  # YouTube link, default concepts, image mode
  python main.py "https://www.youtube.com/watch?v=XXXX" -o out.mp4

  # Local file, only workers and machinery
  python main.py site.mp4 -o out.mp4 --concepts workers machinery

  # First 300 frames only (quick test)
  python main.py site.mp4 --max-frames 300

  # Use the built-in demo YouTube clip
  python main.py --demo -o demo_out.mp4
"""

import argparse
import logging
import sys

from src.config import DEFAULT_CONCEPTS, CONCEPT_GROUPS, DEMO_YOUTUBE_URL


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect workers, machinery and license plates on a "
                    "construction site using Meta's SAM 3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", nargs="?",
                   help="Local video file path OR YouTube URL.")
    p.add_argument("--demo", action="store_true",
                   help=f"Use the built-in demo YouTube video ({DEMO_YOUTUBE_URL}).")
    p.add_argument("-o", "--output", default="output.mp4",
                   help="Output annotated video path (default: output.mp4).")
    p.add_argument("--concepts", nargs="+", default=DEFAULT_CONCEPTS,
                   choices=list(CONCEPT_GROUPS.keys()),
                   help=f"Concept groups to detect (default: {DEFAULT_CONCEPTS}).")
    p.add_argument("--device", default="auto",
                   help="auto | cuda | mps | cpu (default: auto).")
    p.add_argument("--score-thresh", type=float, default=0.5,
                   help="Minimum object score to keep (default: 0.5).")
    p.add_argument("--mask-thresh", type=float, default=0.5,
                   help="Mask binarization threshold (default: 0.5).")
    p.add_argument("--batch", type=int, default=1,
                   help="Frames per model pass (default 1). Higher = more GPU "
                        "utilization/VRAM and faster on GPU. Try 8-16 on a 15GB GPU.")
    p.add_argument("--fp16", action="store_true",
                   help="Load model in half precision (CUDA only) to save VRAM.")
    p.add_argument("--image-size", type=int, default=None,
                   help="Working resolution, e.g. 560 (default 1008). "
                        "Lower = less VRAM but lower accuracy. Helps small GPUs.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Process at most N frames (useful for quick tests).")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Process every Nth frame (default: 1 = every frame).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    source = DEMO_YOUTUBE_URL if args.demo else args.source
    if not source:
        print("ERROR: provide a video file / YouTube URL, or use --demo.\n",
              file=sys.stderr)
        build_parser().print_help()
        return 2

    try:
        from src.pipeline import run  # lazy import (pulls in cv2 / torch)
        out = run(
            source=source,
            output_path=args.output,
            concepts=args.concepts,
            device=args.device,
            score_thresh=args.score_thresh,
            mask_thresh=args.mask_thresh,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            fp16=args.fp16,
            image_size=args.image_size,
            batch=args.batch,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        logging.error(f"Failed: {e}")
        if args.verbose:
            raise
        return 1

    print(f"\n✅ Annotated video written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
