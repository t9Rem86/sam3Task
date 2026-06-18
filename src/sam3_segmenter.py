"""
SAM 3 concept segmenter via 🤗 Transformers.

Follows the official HuggingFace docs:
    https://huggingface.co/docs/transformers/model_doc/sam3

SAM 3 performs *Promptable Concept Segmentation* (PCS) on images: given a short
text phrase (e.g. "construction worker") it returns instance masks + boxes +
scores for **every** matching object. We run one prompt per concept phrase on
each video frame and aggregate the results.

API used (verbatim from the docs):

    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    inputs = processor(images=image, text="ear", return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist())[0]
    # results -> {"masks": ..., "boxes": (xyxy px), "scores": ...}

To avoid recomputing the (expensive) vision backbone for every concept phrase on
the same frame, we precompute vision embeddings once per frame via
`model.get_vision_features(...)` and reuse them across prompts, exactly as shown
in the "Efficient Multi-Prompt Inference on Single Image" section of the docs.
"""

from __future__ import annotations

import logging
import numpy as np

from .config import CONCEPT_GROUPS

logger = logging.getLogger(__name__)

SAM3_CHECKPOINT = "facebook/sam3"


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _pick_device(device: str) -> str:
    import torch
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Sam3ConceptSegmenter:
    """Per-frame SAM 3 segmentation over a set of concept groups."""

    def __init__(
        self,
        concept_keys: list[str],
        score_thresh: float = 0.5,
        mask_thresh: float = 0.5,
        device: str = "auto",
        checkpoint: str = SAM3_CHECKPOINT,
        fp16: bool = False,
        image_size: int | None = None,
    ):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = _pick_device(device)
        self.score_thresh = score_thresh
        self.mask_thresh = mask_thresh
        self.concept_keys = concept_keys

        # fp16 only makes sense on CUDA; it ~halves the VRAM footprint.
        self.dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32

        logger.info(
            f"Loading SAM 3 ('{checkpoint}') on device='{self.device}', "
            f"dtype={self.dtype}, image_size={image_size or 'default(1008)'}..."
        )

        # Optionally shrink the working resolution to fit small GPUs (the model
        # is meant for 1008px; lower = less VRAM but lower accuracy).
        model_kwargs = {}
        proc_kwargs = {}
        if image_size:
            from transformers import Sam3Config
            config = Sam3Config.from_pretrained(checkpoint)
            config.image_size = image_size
            model_kwargs["config"] = config
            proc_kwargs["size"] = {"height": image_size, "width": image_size}

        # Load weights normally then .to(device). We avoid device_map="auto"
        # because accelerate can leave buffers on the "meta" device, crashing
        # with "Tensor.item() cannot be called on meta tensors".
        try:
            self.model = Sam3Model.from_pretrained(
                checkpoint, dtype=self.dtype, **model_kwargs
            )
        except TypeError:  # older transformers used torch_dtype
            self.model = Sam3Model.from_pretrained(
                checkpoint, torch_dtype=self.dtype, **model_kwargs
            )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(checkpoint, **proc_kwargs)

        # All concept phrases flattened, each tagged with its group.
        self.prompts: list[tuple[str, dict]] = [
            (phrase, CONCEPT_GROUPS[key])
            for key in concept_keys
            for phrase in CONCEPT_GROUPS[key]["prompts"]
        ]

    # ── per-frame inference ────────────────────────────────────────────────

    def segment_frame(self, frame_rgb: np.ndarray) -> list[dict]:
        from PIL import Image

        image = Image.fromarray(frame_rgb)

        # Compute vision features once for this frame, reuse across all prompts.
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        target_sizes = img_inputs.get("original_sizes").tolist()

        vision_embeds = None
        try:
            with self.torch.no_grad():
                vision_embeds = self.model.get_vision_features(
                    pixel_values=self._cast_pixels(img_inputs.pixel_values)
                )
        except Exception as e:  # fall back to full forward per prompt
            logger.debug(f"get_vision_features unavailable ({e}); using full forward.")

        detections: list[dict] = []
        for phrase, group in self.prompts:
            results = self._run_prompt(phrase, image, vision_embeds, target_sizes)
            detections.extend(self._to_detections(results, group))

        return _dedupe(detections)

    def segment_frames(self, frames_rgb: list[np.ndarray]) -> list[list[dict]]:
        """Segment a *batch* of frames in one model pass per concept phrase.

        Much higher GPU utilisation than segment_frame (which does one frame at
        a time). Returns one detection list per input frame, in order.
        """
        from PIL import Image

        if not frames_rgb:
            return []

        pil = [Image.fromarray(f) for f in frames_rgb]
        per_image: list[list[dict]] = [[] for _ in pil]

        for phrase, group in self.prompts:
            inputs = self.processor(
                images=pil, text=[phrase] * len(pil), return_tensors="pt"
            ).to(self.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = self._cast_pixels(inputs["pixel_values"])
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.score_thresh,
                mask_threshold=self.mask_thresh,
                target_sizes=inputs.get("original_sizes").tolist(),
            )
            for i, r in enumerate(results):
                per_image[i].extend(self._to_detections(r, group))

        return [_dedupe(dets) for dets in per_image]

    def _cast_pixels(self, pixel_values):
        """Cast pixel_values to the model dtype (e.g. fp16) without touching
        integer tensors like input_ids."""
        if pixel_values is not None and pixel_values.dtype != self.dtype:
            return pixel_values.to(self.dtype)
        return pixel_values

    def _run_prompt(self, phrase, image, vision_embeds, target_sizes) -> dict:
        with self.torch.no_grad():
            if vision_embeds is not None:
                text_inputs = self.processor(text=phrase, return_tensors="pt").to(self.device)
                outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
            else:
                inputs = self.processor(
                    images=image, text=phrase, return_tensors="pt"
                ).to(self.device)
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = self._cast_pixels(inputs["pixel_values"])
                outputs = self.model(**inputs)

        return self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_thresh,
            mask_threshold=self.mask_thresh,
            target_sizes=target_sizes,
        )[0]

    def _to_detections(self, results: dict, group: dict) -> list[dict]:
        masks = _to_numpy(results.get("masks"))
        boxes = _to_numpy(results.get("boxes"))
        scores = _to_numpy(results.get("scores"))
        if boxes is None or len(boxes) == 0:
            return []

        dets = []
        for i in range(len(boxes)):
            score = float(scores[i]) if scores is not None else 1.0
            mask = None
            if masks is not None and i < len(masks):
                mask = np.asarray(masks[i]).astype(bool)
            dets.append({
                "box": np.asarray(boxes[i]).astype(int),
                "mask": mask,
                "score": score,
                "category": group["label"],
                "color": group["color"],
                "track_id": -1,
            })
        return dets


# ── duplicate suppression (multiple phrases per category may overlap) ────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _dedupe(dets: list[dict], iou_thresh: float = 0.7) -> list[dict]:
    """Greedy NMS within the same category across overlapping phrases."""
    out: list[dict] = []
    for d in sorted(dets, key=lambda x: x["score"], reverse=True):
        if any(d["category"] == k["category"] and _iou(d["box"], k["box"]) > iou_thresh
               for k in out):
            continue
        out.append(d)
    return out
