from __future__ import annotations

import base64
import json
import math
import os
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
from PIL import Image


# Sampling config
FRAME_INTERVAL = 30

# Output config
PROCESS_TOTAL = 1
CLEAN_ORPHAN_OUTPUTS_ON_START = 1

# OmniParser config
BOX_THRESHOLD = 0.05

# Region layout config
TOP_BAR_HEIGHT_RATIO = 0.14
LEFT_TOOLBAR_WIDTH_RATIO = 0.12
RIGHT_PANEL_START_RATIO = 0.76
BOTTOM_NOISE_START_RATIO = 0.93
SUBTITLE_START_RATIO = 0.70

# Matching config
MATCH_SCORE_THRESHOLD = 0.58
MODIFIED_IOU_THRESHOLD = 0.70
MODIFIED_CENTER_SHIFT_RATIO = 0.03
TEXT_MATCH_LENGTH_LIMIT = 20

# Decision config
HIGH_CONFIDENCE_THRESHOLD = 0.72
TOTAL_CHANGE_KEEP_THRESHOLD = 5.00
SIGNIFICANT_CHANGE_SCORE_THRESHOLD = 0.45
SIGNIFICANT_CHANGE_COUNT_THRESHOLD = 3
HIGH_CONF_KEEP_SCORE_THRESHOLD = 0.85

# Weight config
CHANGE_TYPE_WEIGHTS = {
    "added": 1.00,
    "removed": 1.00,
    "modified": 0.70,
}

TYPE_WEIGHTS = {
    "icon": 1.00,
    "text": 0.45,
}

REGION_WEIGHTS = {
    "top_bar": 1.20,
    "left_toolbar": 1.30,
    "right_panel": 1.20,
    "canvas_overlay": 1.10,
    "other": 0.20,
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMPORTANT_REGIONS = {"top_bar", "left_toolbar", "right_panel", "canvas_overlay"}
KEEP_HISTORY_SIZE = 2

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSING_ROOT_DIR = ROOT_DIR / "processing"
VIDEO_UNLABEL_DIR = ROOT_DIR / "video" / "unlabel"
PROCESS_ROOT_DIR = ROOT_DIR / "process"
PROCESS_IMAGE_DIR = PROCESS_ROOT_DIR / "ima"
PROCESS_DATA_DIR = PROCESS_ROOT_DIR / "data"
PROCESS_TOTAL_DIR = PROCESS_ROOT_DIR / "total"
FINAL_ROOT_DIR = PROCESSING_ROOT_DIR / "1.final_unlabel"
FINAL_IMAGE_DIR = FINAL_ROOT_DIR / "ima"
FINAL_DATA_DIR = FINAL_ROOT_DIR / "data"
FINAL_TOTAL_DIR = FINAL_ROOT_DIR / "total"
OMNIPARSER_DIR = ROOT_DIR / "OmniParser"


@dataclass(frozen=True)
class MatchResult:
    previous_index: int
    current_index: int
    score: float


def ensure_dirs() -> None:
    VIDEO_UNLABEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PROCESS_TOTAL:
        PROCESS_TOTAL_DIR.mkdir(parents=True, exist_ok=True)
        FINAL_TOTAL_DIR.mkdir(parents=True, exist_ok=True)
    if CLEAN_ORPHAN_OUTPUTS_ON_START:
        cleanup_orphan_outputs()


def ensure_runtime_env() -> None:
    (OMNIPARSER_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)
    (OMNIPARSER_DIR / ".hf").mkdir(parents=True, exist_ok=True)
    os_env = {
        "ULTRALYTICS_SETTINGS_DIR": str(OMNIPARSER_DIR / ".ultralytics"),
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "EASYOCR_MODULE_PATH": str(OMNIPARSER_DIR / ".easyocr"),
        "HF_HOME": str(OMNIPARSER_DIR / ".hf"),
        "HUGGINGFACE_HUB_CACHE": str(OMNIPARSER_DIR / ".hf" / "hub"),
    }
    for key, value in os_env.items():
        if key not in os.environ:
            os.environ[key] = value


def find_first_video() -> Path | None:
    videos = sorted(
        path for path in VIDEO_UNLABEL_DIR.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos[0] if videos else None


def get_frame_image_path(image_name: str) -> Path:
    return PROCESS_IMAGE_DIR / image_name


def get_frame_json_path(image_name: str) -> Path:
    return PROCESS_DATA_DIR / f"{Path(image_name).stem}_omniparser.json"


def get_frame_diff_path(image_name: str) -> Path:
    return PROCESS_DATA_DIR / f"{Path(image_name).stem}_diff.json"


def get_frame_total_path(image_name: str) -> Path:
    return PROCESS_TOTAL_DIR / f"{Path(image_name).stem}_overlay.png"


def get_final_image_path(image_name: str) -> Path:
    return FINAL_IMAGE_DIR / image_name


def get_final_json_path(image_name: str) -> Path:
    return FINAL_DATA_DIR / f"{Path(image_name).stem}_omniparser.json"


def get_final_diff_path(image_name: str) -> Path:
    return FINAL_DATA_DIR / f"{Path(image_name).stem}_diff.json"


def get_final_total_path(image_name: str) -> Path:
    return FINAL_TOTAL_DIR / f"{Path(image_name).stem}_overlay.png"


def cleanup_outputs(
    image_path: Path | None = None,
    json_path: Path | None = None,
    diff_path: Path | None = None,
    overlay_path: Path | None = None,
) -> None:
    for path in (image_path, json_path, diff_path, overlay_path):
        if path is not None:
            path.unlink(missing_ok=True)


def move_if_exists(source: Path | None, target: Path | None) -> Path | None:
    if source is None or target is None or not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    source.replace(target)
    return target


def finalize_keep(
    image_name: str,
    image_path: Path,
    json_path: Path,
    overlay_path: Path | None = None,
) -> dict[str, Path | None]:
    final_image_path = move_if_exists(image_path, get_final_image_path(image_name))
    final_json_path = move_if_exists(json_path, get_final_json_path(image_name))
    final_total_path = move_if_exists(overlay_path, get_final_total_path(image_name)) if overlay_path is not None else None
    return {
        "image_path": final_image_path,
        "json_path": final_json_path,
        "overlay_path": final_total_path,
    }


def cleanup_orphan_outputs() -> int:
    removed_count = 0
    for image_path in PROCESS_IMAGE_DIR.glob("*.png"):
        json_path = get_frame_json_path(image_path.name)
        if json_path.exists():
            continue
        cleanup_outputs(
            image_path=image_path,
            diff_path=get_frame_diff_path(image_path.name),
            overlay_path=get_frame_total_path(image_path.name) if PROCESS_TOTAL_DIR.exists() else None,
        )
        removed_count += 1
    for json_path in PROCESS_DATA_DIR.glob("*_omniparser.json"):
        image_name = f"{json_path.stem.removesuffix('_omniparser')}.png"
        image_path = get_frame_image_path(image_name)
        if image_path.exists():
            continue
        cleanup_outputs(
            json_path=json_path,
            diff_path=get_frame_diff_path(image_name),
            overlay_path=get_frame_total_path(image_name) if PROCESS_TOTAL_DIR.exists() else None,
        )
        removed_count += 1
    if removed_count:
        print(f"Cleaned orphan outputs: {removed_count}")
    return removed_count


def get_omniparser():
    ensure_runtime_env()
    omni_dir = str(OMNIPARSER_DIR)
    if omni_dir not in sys.path:
        sys.path.insert(0, omni_dir)

    from util.omniparser import Omniparser

    return Omniparser(
        {
            "som_model_path": str(OMNIPARSER_DIR / "weights" / "icon_detect" / "model.pt"),
            "caption_model_name": "florence2",
            "caption_model_path": str(OMNIPARSER_DIR / "weights" / "icon_caption_florence"),
            "BOX_TRESHOLD": BOX_THRESHOLD,
        }
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split()).lower()


def ratio_bbox_to_pixels(bbox: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    pixel_bbox = [
        int(round(clamp(x1, 0.0, 1.0) * width)),
        int(round(clamp(y1, 0.0, 1.0) * height)),
        int(round(clamp(x2, 0.0, 1.0) * width)),
        int(round(clamp(y2, 0.0, 1.0) * height)),
    ]
    pixel_bbox[2] = max(pixel_bbox[0] + 1, min(width, pixel_bbox[2]))
    pixel_bbox[3] = max(pixel_bbox[1] + 1, min(height, pixel_bbox[3]))
    return pixel_bbox


def bbox_center(bbox: list[int]) -> list[int]:
    return [int(round((bbox[0] + bbox[2]) / 2)), int(round((bbox[1] + bbox[3]) / 2))]


def bbox_size_ratios(bbox: list[int], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height


def classify_region(bbox: list[int], width: int, height: int) -> str:
    x_ratio, y_ratio, w_ratio, h_ratio = bbox_size_ratios(bbox, width, height)
    x2_ratio = bbox[2] / width
    y2_ratio = bbox[3] / height

    if y2_ratio <= TOP_BAR_HEIGHT_RATIO:
        return "top_bar"
    if x2_ratio <= LEFT_TOOLBAR_WIDTH_RATIO and y_ratio >= 0.04 and y2_ratio <= 0.94:
        return "left_toolbar"
    if x_ratio >= RIGHT_PANEL_START_RATIO and y_ratio >= 0.08 and y2_ratio <= 0.94:
        return "right_panel"
    if 0.10 <= x_ratio <= 0.90 and 0.12 <= y_ratio <= 0.92 and w_ratio >= 0.08 and h_ratio >= 0.08:
        return "canvas_overlay"
    return "other"


def noise_reason(name: str, raw_type: str, bbox: list[int], width: int, height: int) -> str | None:
    _, y_ratio, w_ratio, h_ratio = bbox_size_ratios(bbox, width, height)

    if y_ratio >= BOTTOM_NOISE_START_RATIO:
        return "bottom_system_or_player_band"
    if raw_type == "text" and y_ratio >= SUBTITLE_START_RATIO and w_ratio >= 0.25:
        return "subtitle_overlay"
    if raw_type == "text" and y_ratio <= 0.08 and w_ratio >= 0.20:
        return "video_title_or_banner"
    if raw_type == "text" and len(name) > 28 and w_ratio >= 0.18:
        return "long_text_overlay"
    if raw_type == "icon" and y_ratio >= 0.88 and h_ratio <= 0.08:
        return "player_control_icon"
    if "youtube.com" in name or "press esc" in name or " esc " in f" {name} ":
        return "fullscreen_hint"
    return None


def build_element(raw_element: dict[str, Any], width: int, height: int, element_id: int) -> dict[str, Any] | None:
    bbox_ratio = [float(value) for value in raw_element.get("bbox", [0.0, 0.0, 0.0, 0.0])]
    bbox = ratio_bbox_to_pixels(bbox_ratio, width, height)
    name = normalize_text(raw_element.get("content"))
    raw_type = normalize_text(raw_element.get("type")) or "other"
    confidence = clamp(float(raw_element.get("confidence", 0.0)), 0.0, 1.0)
    clickable = bool(raw_element.get("interactivity", raw_type == "icon"))
    filtered_reason = noise_reason(name, raw_type, bbox, width, height)
    if filtered_reason:
        return None

    region = classify_region(bbox, width, height)
    return {
        "id": str(element_id),
        "name": name,
        "bbox": bbox,
        "bbox_ratio": [round(value, 6) for value in bbox_ratio],
        "center": bbox_center(bbox),
        "raw_type": raw_type,
        "clickable": clickable,
        "confidence": round(confidence, 4),
        "region": region,
        "source": raw_element.get("source", "omniparser"),
    }


def parse_frame(image_path: Path, parser, image_name: str | None = None) -> tuple[dict[str, Any], Path, Path | None]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    overlay_b64, raw_elements = parser.parse(image_b64)

    with Image.open(image_path) as image:
        width, height = image.size

    filtered_elements: list[dict[str, Any]] = []
    for index, raw_element in enumerate(raw_elements, start=1):
        element = build_element(raw_element, width, height, index)
        if element:
            filtered_elements.append(element)

    parsed_json = {
        "image": str(image_path),
        "image_size": [width, height],
        "raw_element_count": len(raw_elements),
        "element_count": len(filtered_elements),
        "elements": filtered_elements,
    }

    image_name = image_name or image_path.name
    json_path = get_frame_json_path(image_name)
    json_path.write_text(json.dumps(parsed_json, indent=2, ensure_ascii=False), encoding="utf-8")

    overlay_path: Path | None = None
    if PROCESS_TOTAL:
        overlay_path = get_frame_total_path(image_name)
        overlay_path.write_bytes(base64.b64decode(overlay_b64))

    return parsed_json, json_path, overlay_path


def bbox_iou(box1: list[int], box2: list[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - intersection
    return intersection / max(union, 1)


def center_distance_ratio(box1: list[int], box2: list[int], width: int, height: int) -> float:
    center1 = bbox_center(box1)
    center2 = bbox_center(box2)
    distance = math.dist(center1, center2)
    diagonal = math.hypot(width, height)
    return distance / max(diagonal, 1.0)


def text_similarity(text1: str, text2: str) -> float:
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1[:TEXT_MATCH_LENGTH_LIMIT], text2[:TEXT_MATCH_LENGTH_LIMIT]).ratio()


def compute_match_score(previous: dict[str, Any], current: dict[str, Any], width: int, height: int) -> float:
    iou_score = bbox_iou(previous["bbox"], current["bbox"])
    distance_score = max(0.0, 1.0 - center_distance_ratio(previous["bbox"], current["bbox"], width, height) / 0.08)
    text_score = text_similarity(previous["name"], current["name"])
    type_pair = {previous["raw_type"], current["raw_type"]}

    if previous["raw_type"] == current["raw_type"]:
        type_score = 1.0
    elif type_pair == {"text", "icon"}:
        # OmniParser sometimes flips the same UI element between OCR text and icon caption.
        # Only allow this when the geometry is very close.
        if iou_score < 0.65 and distance_score < 0.92:
            return 0.0
        type_score = 0.60
    else:
        return 0.0

    if previous["region"] == current["region"]:
        region_score = 1.0
    elif "other" in {previous["region"], current["region"]}:
        region_score = 0.35
    else:
        region_score = 0.0

    score = ((0.55 * iou_score) + (0.20 * distance_score) + (0.15 * text_score) + (0.10 * region_score)) * type_score
    if previous["clickable"] != current["clickable"]:
        score -= 0.05
    return max(0.0, score)


def match_elements(
    previous_elements: list[dict[str, Any]],
    current_elements: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[list[MatchResult], set[int], set[int]]:
    candidate_matches: list[MatchResult] = []
    for previous_index, previous_element in enumerate(previous_elements):
        for current_index, current_element in enumerate(current_elements):
            score = compute_match_score(previous_element, current_element, width, height)
            if score >= MATCH_SCORE_THRESHOLD:
                candidate_matches.append(MatchResult(previous_index, current_index, score))

    matched_previous: set[int] = set()
    matched_current: set[int] = set()
    final_matches: list[MatchResult] = []
    for candidate in sorted(candidate_matches, key=lambda item: item.score, reverse=True):
        if candidate.previous_index in matched_previous or candidate.current_index in matched_current:
            continue
        matched_previous.add(candidate.previous_index)
        matched_current.add(candidate.current_index)
        final_matches.append(candidate)

    unmatched_previous = set(range(len(previous_elements))) - matched_previous
    unmatched_current = set(range(len(current_elements))) - matched_current
    return final_matches, unmatched_previous, unmatched_current


def is_modified(previous: dict[str, Any], current: dict[str, Any], width: int, height: int) -> bool:
    if previous["region"] != current["region"]:
        return True
    if previous["raw_type"] != current["raw_type"]:
        return True
    if bbox_iou(previous["bbox"], current["bbox"]) < MODIFIED_IOU_THRESHOLD:
        return True
    if center_distance_ratio(previous["bbox"], current["bbox"], width, height) > MODIFIED_CENTER_SHIFT_RATIO:
        return True
    return False


def change_confidence(change_type: str, previous: dict[str, Any] | None, current: dict[str, Any] | None) -> float:
    if change_type == "removed" and previous is not None:
        return float(previous["confidence"])
    if current is not None:
        return float(current["confidence"])
    return 0.0


def base_weight(element: dict[str, Any]) -> float:
    type_weight = TYPE_WEIGHTS.get(element["raw_type"], 0.35)
    region_weight = REGION_WEIGHTS.get(element["region"], REGION_WEIGHTS["other"])
    interaction_weight = 1.0 if element["clickable"] else 0.60
    return type_weight * region_weight * interaction_weight


def event_score(change_type: str, previous: dict[str, Any] | None, current: dict[str, Any] | None) -> float:
    anchor = current or previous
    if anchor is None:
        return 0.0
    confidence = max(change_confidence(change_type, previous, current), 0.0)
    return CHANGE_TYPE_WEIGHTS[change_type] * base_weight(anchor) * confidence


def serialize_event(
    change_type: str,
    previous: dict[str, Any] | None,
    current: dict[str, Any] | None,
    score: float,
) -> dict[str, Any]:
    anchor = current or previous or {}
    return {
        "change_type": change_type,
        "score": round(score, 4),
        "confidence": round(change_confidence(change_type, previous, current), 4),
        "region": anchor.get("region", "other"),
        "raw_type": anchor.get("raw_type", "other"),
        "clickable": anchor.get("clickable", False),
        "name": anchor.get("name", ""),
        "previous": previous,
        "current": current,
    }


def summarize_diff(previous_state: dict[str, Any], current_state: dict[str, Any]) -> dict[str, Any]:
    width, height = current_state["image_size"]
    previous_elements = previous_state["elements"]
    current_elements = current_state["elements"]
    matches, unmatched_previous, unmatched_current = match_elements(previous_elements, current_elements, width, height)

    added_events: list[dict[str, Any]] = []
    removed_events: list[dict[str, Any]] = []
    modified_events: list[dict[str, Any]] = []

    for current_index in sorted(unmatched_current):
        current_element = current_elements[current_index]
        added_events.append(
            serialize_event("added", None, current_element, event_score("added", None, current_element))
        )

    for previous_index in sorted(unmatched_previous):
        previous_element = previous_elements[previous_index]
        removed_events.append(
            serialize_event("removed", previous_element, None, event_score("removed", previous_element, None))
        )

    for match in matches:
        previous_element = previous_elements[match.previous_index]
        current_element = current_elements[match.current_index]
        if is_modified(previous_element, current_element, width, height):
            modified_events.append(
                serialize_event(
                    "modified",
                    previous_element,
                    current_element,
                    event_score("modified", previous_element, current_element),
                )
            )

    all_events = added_events + removed_events + modified_events
    total_change_score = sum(event["score"] for event in all_events)
    significant_change_count = sum(event["score"] >= SIGNIFICANT_CHANGE_SCORE_THRESHOLD for event in all_events)
    high_conf_events = [
        event
        for event in all_events
        if event["confidence"] >= HIGH_CONFIDENCE_THRESHOLD and event["region"] in IMPORTANT_REGIONS
    ]
    required_high_conf_events = [
        event
        for event in all_events
        if event["confidence"] >= HIGH_CONFIDENCE_THRESHOLD
        and event["region"] != "canvas_overlay"
        and event["raw_type"] != "text"
    ]
    high_conf_change_count = len(high_conf_events)
    high_conf_change_score = sum(event["score"] for event in high_conf_events)
    required_high_conf_change_count = len(required_high_conf_events)
    important_new_events = [
        event
        for event in added_events
        if event["region"] in IMPORTANT_REGIONS and event["confidence"] >= HIGH_CONFIDENCE_THRESHOLD
    ]

    keep = False
    keep_reasons: list[str] = []
    if required_high_conf_change_count >= 1:
        if total_change_score >= TOTAL_CHANGE_KEEP_THRESHOLD:
            keep = True
            keep_reasons.append("total_change_score")
        if significant_change_count >= SIGNIFICANT_CHANGE_COUNT_THRESHOLD:
            keep = True
            keep_reasons.append("significant_change_count")
        if high_conf_change_score >= HIGH_CONF_KEEP_SCORE_THRESHOLD:
            keep = True
            keep_reasons.append("high_conf_change")
        if important_new_events:
            keep = True
            keep_reasons.append("important_new_element")

    return {
        "previous_image": previous_state["image"],
        "current_image": current_state["image"],
        "summary": {
            "added_count": len(added_events),
            "removed_count": len(removed_events),
            "modified_count": len(modified_events),
            "total_change_score": round(total_change_score, 4),
            "significant_change_count": significant_change_count,
            "high_conf_change_count": high_conf_change_count,
            "high_conf_change_score": round(high_conf_change_score, 4),
            "required_high_conf_change_count": required_high_conf_change_count,
            "keep": keep,
            "keep_reasons": keep_reasons,
        },
        "added": added_events,
        "removed": removed_events,
        "modified": modified_events,
    }


def compare_against_recent_kept(
    recent_kept_states: list[dict[str, Any]],
    current_state: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    for history_offset, previous_state in enumerate(reversed(recent_kept_states[-KEEP_HISTORY_SIZE:]), start=1):
        diff_data = summarize_diff(previous_state, current_state)
        summary = diff_data["summary"]
        comparisons.append(
            {
                "history_offset": history_offset,
                "keep": bool(summary["keep"]),
                "added_count": int(summary["added_count"]),
                "removed_count": int(summary["removed_count"]),
                "modified_count": int(summary["modified_count"]),
                "total_change_score": float(summary["total_change_score"]),
                "high_conf_change_count": int(summary["high_conf_change_count"]),
                "keep_reasons": list(summary["keep_reasons"]),
            }
        )

    should_keep = all(item["keep"] for item in comparisons) if comparisons else True
    return should_keep, comparisons


def extend_kept_history(
    recent_kept_states: list[dict[str, Any]],
    current_state: dict[str, Any],
) -> list[dict[str, Any]]:
    return [*recent_kept_states[-(KEEP_HISTORY_SIZE - 1) :], current_state]


def format_keep_log(comparisons: list[dict[str, Any]]) -> str:
    if not comparisons:
        return "first sampled frame"

    return " | ".join(
        (
            f"prev{item['history_offset']}:score={item['total_change_score']:.3f} "
            f"high_conf={item['high_conf_change_count']} "
            f"reasons={','.join(item['keep_reasons']) or 'none'}"
        )
        for item in comparisons
    )


def format_drop_log(comparisons: list[dict[str, Any]]) -> str:
    if not comparisons:
        return "no previous kept frames"

    return " | ".join(
        (
            f"prev{item['history_offset']}:keep={str(item['keep']).lower()} "
            f"score={item['total_change_score']:.3f} "
            f"added={item['added_count']} removed={item['removed_count']} "
            f"modified={item['modified_count']}"
        )
        for item in comparisons
    )


def process_video(video_path: Path, parser) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_index = 0
    sampled_count = 0
    kept_count = 0
    dropped_count = 0
    recent_kept_states: list[dict[str, Any]] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % FRAME_INTERVAL != 0:
            frame_index += 1
            continue

        sampled_count += 1
        timestamp_seconds = frame_index / fps if fps > 0 else 0.0
        image_name = f"{video_path.stem}_{frame_index:06d}_{timestamp_seconds:08.2f}s.png"
        image_path = get_frame_image_path(image_name)
        cv2.imwrite(str(image_path), frame)

        current_state, json_path, overlay_path = parse_frame(image_path, parser, image_name=image_name)

        if not recent_kept_states:
            kept_count += 1
            recent_kept_states = extend_kept_history(recent_kept_states, current_state)
            final_paths = finalize_keep(image_name, image_path, json_path, overlay_path=overlay_path)
            print(
                f"[KEEP] {image_name} first sampled frame "
                f"elements={current_state['element_count']} raw={current_state['raw_element_count']}"
            )
            print(f"         image={final_paths['image_path']}")
            print(f"         json={final_paths['json_path']}")
            if final_paths["overlay_path"] is not None:
                print(f"         total={final_paths['overlay_path']}")
        else:
            should_keep, comparisons = compare_against_recent_kept(recent_kept_states, current_state)
            if should_keep:
                kept_count += 1
                recent_kept_states = extend_kept_history(recent_kept_states, current_state)
                final_paths = finalize_keep(image_name, image_path, json_path, overlay_path=overlay_path)
                print(f"[KEEP] {image_name} {format_keep_log(comparisons)}")
                print(f"         image={final_paths['image_path']}")
                print(f"         json={final_paths['json_path']}")
                if final_paths["overlay_path"] is not None:
                    print(f"         total={final_paths['overlay_path']}")
            else:
                cleanup_outputs(
                    image_path=image_path,
                    json_path=json_path,
                    diff_path=get_frame_diff_path(image_name),
                    overlay_path=overlay_path,
                )
                dropped_count += 1
                print(f"[DROP] {image_name} {format_drop_log(comparisons)}")

        frame_index += 1

    capture.release()
    print()
    print(f"Processed video: {video_path.name}")
    print(f"Sampled frames: {sampled_count}")
    print(f"Saved images: {kept_count}")
    print(f"Dropped frames: {dropped_count}")
    print(f"Image output: {FINAL_IMAGE_DIR}")
    print(f"Data output: {FINAL_DATA_DIR}")
    if PROCESS_TOTAL:
        print(f"Total output: {FINAL_TOTAL_DIR}")


def main() -> None:
    ensure_dirs()
    source_video = find_first_video()
    if source_video is None:
        print(f"No video found in {VIDEO_UNLABEL_DIR}")
        return

    parser = get_omniparser()
    print(f"Processing video in place: {source_video}")
    process_video(source_video, parser)


if __name__ == "__main__":
    main()
