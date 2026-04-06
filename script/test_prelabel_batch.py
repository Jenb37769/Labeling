from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont

import prelabel as shared_prelabel


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSING_ROOT_DIR = ROOT_DIR / "processing"
SOURCE_IMAGE_DIR = PROCESSING_ROOT_DIR / "1.final_unlabel" / "ima"
SOURCE_DATA_DIR = PROCESSING_ROOT_DIR / "1.final_unlabel" / "data"
OUTPUT_ROOT_DIR = PROCESSING_ROOT_DIR / "2.prelabel"
OUTPUT_IMAGE_DIR = OUTPUT_ROOT_DIR / "ima"
OUTPUT_DATA_DIR = OUTPUT_ROOT_DIR / "data"
OUTPUT_TOTAL_DIR = OUTPUT_ROOT_DIR / "total"

# Batch config
DEFAULT_ICON_PER_BATCH = 6
DEFAULT_ENABLE_REASONING = True

REQUEST_RETRY_COUNT = 3
REQUEST_RETRY_SLEEP_SECONDS = 2.0
PARSE_RETRY_COUNT = 3
PARSE_RETRY_TOKEN_STEP = 2000

ALLOWED_TYPES = [
    "tool_icon",
    "menu_item",
    "submenu_item",
    "panel_tab",
    "panel_item",
    "icon_button",
    "text_button",
    "dropdown",
    "input_field",
    "slider",
    "toggle",
    "canvas_target",
]

SYSTEM_PROMPT = (
    "You are a professional labeling assistant for Photoshop UI screenshots. "
    "You will receive one overview image plus multiple zoomed crops in the same request. "
    "The overview image contains thin boxes and numeric ids for all candidate targets in the current batch. "
    "Each crop image corresponds to one id from the overview image. "
    "Return exactly one JSON object and classify every provided id."
)


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one complete sample from processing/1.final_unlabel and write label-ready outputs into "
            "processing/2.prelabel/ima, data, and total."
        )
    )
    parser.add_argument("--image-dir", type=Path, default=SOURCE_IMAGE_DIR)
    parser.add_argument("--data-dir", type=Path, default=SOURCE_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DIR)
    parser.add_argument(
        "--stem",
        type=str,
        default="",
        help="Optional frame stem. If omitted, the first complete sample with both JSON and image is used.",
    )
    parser.add_argument("--batch-index", type=int, default=0, help="Zero-based batch index.")
    parser.add_argument(
        "--icon-per-batch",
        "--icon-per-bench",
        dest="icon_per_batch",
        type=int,
        default=DEFAULT_ICON_PER_BATCH,
    )
    parser.add_argument("--limit-batches", type=int, default=1000, help="How many consecutive batches to run.")
    parser.add_argument("--model", type=str, default=os.environ.get("PRELABEL_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", type=str, default=os.environ.get("PRELABEL_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("PRELABEL_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--http-referer", type=str, default=os.environ.get("PRELABEL_HTTP_REFERER", ""))
    parser.add_argument("--app-title", type=str, default=os.environ.get("PRELABEL_APP_TITLE", "grounding-batch-prelabel"))
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        default=DEFAULT_ENABLE_REASONING,
        help="Enable provider reasoning if supported.",
    )
    parser.add_argument(
        "--disable-reasoning",
        action="store_false",
        dest="enable_reasoning",
        help="Disable provider reasoning even if the file default enables it.",
    )
    args = parser.parse_args()
    args.model = str(args.model or "").strip()
    args.base_url = str(args.base_url or "").strip()
    args.api_key = str(args.api_key or "").strip()
    args.http_referer = str(args.http_referer or "").strip()
    args.app_title = str(args.app_title or "").strip()
    args.output_root = Path(args.output_root)
    return args


def ensure_output_dirs(output_root: Path) -> tuple[Path, Path, Path]:
    image_dir = output_root / "ima"
    data_dir = output_root / "data"
    total_dir = output_root / "total"
    image_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    total_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, data_dir, total_dir


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clamp_bbox(bbox: list[Any], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def expand_bbox(bbox: list[int], width: int, height: int, padding_ratio: float = 0.35, min_padding: int = 20) -> list[int]:
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = max(min_padding, int(round(box_width * padding_ratio)))
    pad_y = max(min_padding, int(round(box_height * padding_ratio)))
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    ]


def fit_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", max(10, size))
    except OSError:
        return ImageFont.load_default()


def build_overview_image(image: Image.Image, items: list[dict[str, Any]]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    outline_width = max(1, int(round(max(canvas.size) * 0.0018)))
    font = fit_font(max(10, int(round(max(canvas.size) * 0.012))))

    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        label = str(item["id"])
        draw.rectangle((x1, y1, x2, y2), outline=(255, 48, 48), width=outline_width)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = max(1, text_bbox[2] - text_bbox[0])
        text_height = max(1, text_bbox[3] - text_bbox[1])
        label_x = max(0, x1)
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 6, label_y + text_height + 4),
            fill=(255, 255, 255),
            outline=(255, 48, 48),
            width=1,
        )
        draw.text((label_x + 3, label_y + 2), label, fill=(255, 48, 48), font=font)

    return canvas


def build_crop_image(image: Image.Image, bbox: list[int], item_id: str) -> Image.Image:
    expanded = expand_bbox(bbox, image.width, image.height)
    crop = image.crop(tuple(expanded)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    local_bbox = [
        bbox[0] - expanded[0],
        bbox[1] - expanded[1],
        bbox[2] - expanded[0],
        bbox[3] - expanded[1],
    ]
    outline_width = max(2, int(round(max(crop.size) * 0.01)))
    font = fit_font(max(10, int(round(max(crop.size) * 0.04))))
    draw.rectangle(tuple(local_bbox), outline=(255, 48, 48), width=outline_width)
    draw.rectangle((6, 6, 28, 24), fill=(255, 255, 255), outline=(255, 48, 48), width=1)
    draw.text((11, 8), str(item_id), fill=(255, 48, 48), font=font)
    return crop


def resize_image_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    longest_side = max(image.size)
    if longest_side <= max_side:
        return image
    scale = max_side / max(longest_side, 1)
    return image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )


def image_to_data_url(image: Image.Image, quality: int = 80) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def normalize_type(value: Any) -> str:
    text = "_".join(str(value or "").strip().lower().replace("-", "_").split())
    return text if text in ALLOWED_TYPES else ""


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = strip_code_fences(text)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON: {text[:300]}")
    return json.loads(stripped[start : end + 1])


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def call_openai_compatible_api(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt_text: str,
    image_data_urls: list[str],
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    http_referer: str,
    app_title: str,
    enable_reasoning: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not api_key:
        raise ValueError("Missing PRELABEL_API_KEY or --api-key.")

    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for data_url in image_data_urls:
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    if enable_reasoning:
        payload["reasoning"] = {"enabled": True}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "grounding-batch-prelabel/1.0",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if app_title:
        headers["X-Title"] = app_title

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    session = requests.Session()
    session.trust_env = False
    request_body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(1, REQUEST_RETRY_COUNT + 1):
        try:
            response = session.post(endpoint, data=request_body, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            response_json = response.json()
            raw_text = flatten_message_content(response_json["choices"][0]["message"]["content"])
            return payload, response_json, raw_text
        except Exception as exc:
            last_error = exc
            if attempt >= REQUEST_RETRY_COUNT:
                raise RuntimeError(f"Batch prelabel request failed after {REQUEST_RETRY_COUNT} attempts: {exc}") from exc
            print(
                f"[RETRY] attempt={attempt}/{REQUEST_RETRY_COUNT} failed; retrying in "
                f"{REQUEST_RETRY_SLEEP_SECONDS:.1f}s"
            )
            time.sleep(REQUEST_RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"Batch prelabel request failed: {last_error}")


def request_and_parse_batch(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt_text: str,
    image_data_urls: list[str],
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    http_referer: str,
    app_title: str,
    enable_reasoning: bool,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], int]:
    current_max_tokens = max_tokens
    last_error: Exception | None = None

    for attempt in range(1, PARSE_RETRY_COUNT + 1):
        payload, response_json, raw_text = call_openai_compatible_api(
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt_text=prompt_text,
            image_data_urls=image_data_urls,
            max_tokens=current_max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            http_referer=http_referer,
            app_title=app_title,
            enable_reasoning=enable_reasoning,
        )

        finish_reason = str(response_json.get("choices", [{}])[0].get("finish_reason", "")).strip().lower()
        try:
            parsed = extract_json_object(raw_text)
            return payload, response_json, raw_text, parsed, current_max_tokens
        except Exception as exc:
            last_error = exc
            if attempt >= PARSE_RETRY_COUNT:
                raise RuntimeError(f"Failed to parse batch response after {PARSE_RETRY_COUNT} attempts: {exc}") from exc

            current_max_tokens += PARSE_RETRY_TOKEN_STEP
            reason_text = "finish_reason=length" if finish_reason == "length" else type(exc).__name__
            print(
                f"[PARSE RETRY] attempt={attempt}/{PARSE_RETRY_COUNT} "
                f"reason={reason_text} next_max_tokens={current_max_tokens}"
            )
            time.sleep(REQUEST_RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"Failed to parse batch response: {last_error}")


def pick_complete_sample(data_dir: Path, image_dir: Path, stem: str) -> tuple[Path, Path]:
    if stem:
        json_path = data_dir / f"{stem}_omniparser.json"
        image_path = image_dir / f"{stem}.png"
        if not json_path.exists():
            raise FileNotFoundError(f"JSON not found for stem: {stem}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for stem: {stem}")
        return json_path, image_path

    for json_path in sorted(data_dir.glob("*_omniparser.json")):
        sample_stem = json_path.stem.removesuffix("_omniparser")
        image_path = image_dir / f"{sample_stem}.png"
        if image_path.exists():
            return json_path, image_path

    raise FileNotFoundError(
        f"No complete sample found in {data_dir} and {image_dir}. A complete sample needs both JSON and PNG."
    )


def extract_stem(json_path: Path) -> str:
    return json_path.stem.removesuffix("_omniparser")


def collect_candidates(document: dict[str, Any], image_size: tuple[int, int]) -> list[dict[str, Any]]:
    width, height = image_size
    candidates: list[dict[str, Any]] = []
    for element in document.get("elements", []):
        bbox = element.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        clamped_bbox = clamp_bbox(bbox, width, height)
        candidates.append(
            {
                "id": str(element.get("id", "")),
                "bbox": clamped_bbox,
                "raw_type": str(element.get("raw_type", "")).strip(),
                "region": str(element.get("region", "")).strip(),
                "clickable": bool(element.get("clickable", False)),
                "existing_name": str(element.get("name", "")).strip(),
                "existing_instruction": str(element.get("instruction", "")).strip(),
            }
        )
    return candidates


def build_prompt_text(stem: str, image_size: tuple[int, int], batch_items: list[dict[str, Any]]) -> str:
    lines = [
        "You are a professional labeling assistant. Output exactly one complete JSON object that must strictly follow the schema below.",
        "Do not output anything extra, and do not omit any part of the structured output.",
        "Output exactly one JSON object with this schema:",
        "{",
        '  "items": [',
        "    {",
        '      "id": "candidate_id",',
        '      "validity": "valid" | "invalid" | "uncertain",',
        f'      "type": one of {json.dumps(ALLOWED_TYPES)} or "",',
        '      "name": "english_snake_case_or_empty",',
        '      "clickable": true | false,',
        '      "instruction": "short_english_instruction_or_empty",',
        '      "reason": "short explanation",',
        '      "confidence": 0.0',
        "    }",
        "  ]",
        "}",
        "",
        "Rules:",
        "- Return exactly one item for every candidate id in this batch.",
        "- Keep the same id order as provided below.",
        "- The first image is the full screenshot with all candidate boxes and ids drawn very thinly.",
        "- The following images are zoomed crops, one crop per id, in the same order as the id list.",
        *shared_prelabel.shared_rule_lines(include_clickable_instruction=True),
        "",
        f"frame_stem: {stem}",
        f"image_size: {[image_size[0], image_size[1]]}",
        f"candidate_count: {len(batch_items)}",
        "candidate_ids_and_metadata:",
    ]
    for item in batch_items:
        position_info = shared_prelabel.describe_position_metadata(item, image_size)
        lines.append(
            f"- id={item['id']}, bbox={item['bbox']}, center={position_info['center']}, "
            f"bbox_ratio_xywh={position_info['bbox_ratio']}, center_ratio_xy={position_info['center_ratio']}, "
            f"position_bucket={position_info['position_bucket']}, region={item['region'] or '(empty)'}, "
            f"region_meaning={position_info['region_meaning']}, "
            f"position_screening_hint={position_info['screening_hint']}, "
            f"raw_type={item['raw_type'] or '(empty)'}, clickable={str(item['clickable']).lower()}, "
            f"existing_name={item['existing_name'] or '(empty)'}, "
            f"existing_instruction={item['existing_instruction'] or '(empty)'}"
        )
    return "\n".join(lines)


def normalize_response(parsed: dict[str, Any], batch_items: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = [item["id"] for item in batch_items]
    items = parsed.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Response JSON does not contain an items list.")

    normalized_items: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "")).strip()
        if item_id:
            by_id[item_id] = item

    for expected_id in expected_ids:
        raw = by_id.get(expected_id, {})
        validity = str(raw.get("validity", "")).strip().lower()
        if validity not in {"valid", "invalid", "uncertain"}:
            validity = "uncertain"
        item_type = normalize_type(raw.get("type", ""))
        name = normalize_name(raw.get("name", ""))
        instruction = normalize_name(raw.get("instruction", ""))
        clickable = bool(raw.get("clickable", False))
        if validity != "valid":
            item_type = ""
            name = ""
            instruction = ""
            clickable = False
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        normalized_items.append(
            {
                "id": expected_id,
                "validity": validity,
                "type": item_type,
                "name": name,
                "clickable": clickable,
                "instruction": instruction,
                "reason": str(raw.get("reason", "")).strip(),
                "confidence": round(confidence, 4),
            }
        )

    return {"items": normalized_items}


def apply_batch_result_to_document(
    document: dict[str, Any],
    normalized: dict[str, Any],
    *,
    model: str,
    base_url: str,
    finish_reason: str,
    usage: dict[str, Any],
) -> int:
    by_id = {str(item.get("id", "")).strip(): item for item in normalized.get("items", []) if isinstance(item, dict)}
    updated_count = 0

    for element in document.get("elements", []):
        element_id = str(element.get("id", "")).strip()
        result = by_id.get(element_id)
        if result is None:
            continue

        previous_name = str(element.get("name", ""))
        previous_type = str(element.get("type", ""))
        previous_instruction = str(element.get("instruction", ""))
        element["prelabel"] = {
            "validity": result["validity"],
            "type": result["type"],
            "name": result["name"],
            "clickable": result["clickable"],
            "instruction": result["instruction"],
            "reason": result["reason"],
            "confidence": result["confidence"],
            "model": model,
            "base_url": base_url,
            "finish_reason": finish_reason,
            "usage": usage,
            "previous_name": previous_name,
            "previous_type": previous_type,
            "previous_instruction": previous_instruction,
            "updated_at": utc_now_iso(),
        }
        if result["validity"] == "valid":
            if result["name"]:
                element["name"] = result["name"]
            if result["type"]:
                element["type"] = result["type"]
            element["clickable"] = result["clickable"]
            if result["instruction"]:
                element["instruction"] = result["instruction"]
        updated_count += 1

    document["prelabel_meta"] = {
        "mode": "batch_test",
        "model": model,
        "base_url": base_url,
        "updated_at": utc_now_iso(),
    }
    return updated_count


def bbox_ratio_from_pixels(bbox: list[int], width: int, height: int) -> list[float]:
    return [
        round(bbox[0] / width, 6),
        round(bbox[1] / height, 6),
        round(bbox[2] / width, 6),
        round(bbox[3] / height, 6),
    ]


def bbox_center(bbox: list[int]) -> list[int]:
    return [int(round((bbox[0] + bbox[2]) / 2)), int(round((bbox[1] + bbox[3]) / 2))]


def convert_element_for_label(element: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any] | None:
    prelabel = element.get("prelabel", {})
    if not isinstance(prelabel, dict):
        return None
    if str(prelabel.get("validity", "")).strip().lower() != "valid":
        return None

    width, height = image_size
    bbox = element.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    pixel_bbox = clamp_bbox(bbox, width, height)
    element_type = str(prelabel.get("type", "")).strip() or str(element.get("type", "")).strip() or "icon_button"
    name = str(prelabel.get("name", "")).strip() or str(element.get("name", "")).strip()
    instruction = str(prelabel.get("instruction", "")).strip() or str(element.get("instruction", "")).strip()
    clickable = bool(prelabel.get("clickable", element.get("clickable", False)))

    return {
        "id": str(element.get("id", "")),
        "name": name,
        "bbox": pixel_bbox,
        "bbox_ratio": bbox_ratio_from_pixels(pixel_bbox, width, height),
        "center": bbox_center(pixel_bbox),
        "raw_type": str(element.get("raw_type", "")).strip(),
        "clickable": clickable,
        "instruction": instruction,
        "confidence": float(element.get("confidence", 0.0)),
        "region": str(element.get("region", "")).strip(),
        "source": str(element.get("source", "")).strip() or "batch_prelabel",
        "type": element_type,
    }


def build_label_ready_document(document: dict[str, Any], target_image_name: str) -> dict[str, Any]:
    image_size_raw = document.get("image_size", [1, 1])
    if not isinstance(image_size_raw, list) or len(image_size_raw) != 2:
        image_size_raw = [1, 1]
    width = max(1, int(image_size_raw[0]))
    height = max(1, int(image_size_raw[1]))
    image_size = (width, height)

    converted_elements = [
        converted
        for converted in (convert_element_for_label(element, image_size) for element in document.get("elements", []))
        if converted is not None
    ]

    return {
        "image": target_image_name,
        "image_size": [width, height],
        "raw_element_count": int(document.get("raw_element_count", len(document.get("elements", [])))),
        "element_count": len(converted_elements),
        "elements": converted_elements,
    }


def box_color_for_type(element_type: str) -> tuple[int, int, int]:
    palette = {
        "tool_icon": (38, 93, 155),
        "menu_item": (123, 63, 161),
        "submenu_item": (123, 63, 161),
        "panel_tab": (21, 127, 82),
        "panel_item": (168, 90, 27),
        "icon_button": (200, 74, 47),
        "text_button": (182, 77, 120),
        "dropdown": (93, 83, 170),
        "input_field": (31, 124, 134),
        "slider": (138, 95, 47),
        "toggle": (141, 52, 70),
        "canvas_target": (15, 108, 120),
    }
    return palette.get(element_type, (200, 74, 47))


def render_total_preview(source_image_path: Path, label_ready_document: dict[str, Any], target_total_path: Path) -> None:
    with Image.open(source_image_path) as image:
        canvas = image.convert("RGBA")

    draw = ImageDraw.Draw(canvas, "RGBA")
    outline_width = max(2, int(round(max(canvas.size) * 0.003)))

    for element in label_ready_document.get("elements", []):
        bbox = element.get("bbox", [0, 0, 1, 1])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in bbox]
        color = box_color_for_type(str(element.get("type", "")).strip())
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=outline_width)

    target_total_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(target_total_path)


def write_prelabel_outputs(
    *,
    source_image_path: Path,
    target_image_path: Path,
    target_json_path: Path,
    target_total_path: Path,
    aggregate_document: dict[str, Any],
) -> dict[str, Any]:
    shutil.copy2(source_image_path, target_image_path)
    label_ready_document = build_label_ready_document(aggregate_document, target_image_path.name)
    write_json(target_json_path, label_ready_document)
    render_total_preview(source_image_path, label_ready_document, target_total_path)
    return label_ready_document


def main() -> None:
    args = parse_args()
    output_image_dir, output_data_dir, output_total_dir = ensure_output_dirs(args.output_root)
    json_path, image_path = pick_complete_sample(args.data_dir, args.image_dir, args.stem)
    stem = extract_stem(json_path)
    source_document = load_json(json_path)

    target_image_path = output_image_dir / f"{stem}.png"
    target_json_path = output_data_dir / f"{stem}_omniparser.json"
    target_total_path = output_total_dir / f"{stem}_overlay.png"

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        candidates = collect_candidates(source_document, image.size)
        if not candidates:
            raise ValueError(f"No bbox candidates found in {json_path}")

        aggregate_document = json.loads(json.dumps(source_document))
        batch_size = max(1, int(args.icon_per_batch))
        start_index = max(0, int(args.batch_index)) * batch_size
        max_batches = max(1, int(args.limit_batches))
        processed_batches = 0
        label_ready_document = write_prelabel_outputs(
            source_image_path=image_path,
            target_image_path=target_image_path,
            target_json_path=target_json_path,
            target_total_path=target_total_path,
            aggregate_document=aggregate_document,
        )

        interrupted = False
        try:
            for local_batch_index in range(max_batches):
                batch_start = start_index + local_batch_index * batch_size
                batch_items = candidates[batch_start : batch_start + batch_size]
                if not batch_items:
                    break

                overview_image = resize_image_to_max_side(build_overview_image(image, batch_items), max_side=1600)
                image_data_urls = [image_to_data_url(overview_image, quality=78)]
                for item in batch_items:
                    crop_image = resize_image_to_max_side(build_crop_image(image, item["bbox"], item["id"]), max_side=768)
                    image_data_urls.append(image_to_data_url(crop_image, quality=82))

                prompt_text = build_prompt_text(stem, image.size, batch_items)
                _, response_json, _, parsed, used_max_tokens = request_and_parse_batch(
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=args.model,
                    prompt_text=prompt_text,
                    image_data_urls=image_data_urls,
                    max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_seconds=args.timeout_seconds,
                http_referer=args.http_referer,
                app_title=args.app_title,
                enable_reasoning=args.enable_reasoning,
            )

                normalized = normalize_response(parsed, batch_items)
                finish_reason = str(response_json.get("choices", [{}])[0].get("finish_reason", ""))
                usage = response_json.get("usage", {})
                applied_count = apply_batch_result_to_document(
                    aggregate_document,
                    normalized,
                    model=args.model,
                    base_url=args.base_url,
                    finish_reason=finish_reason,
                    usage=usage if isinstance(usage, dict) else {},
                )
                label_ready_document = write_prelabel_outputs(
                    source_image_path=image_path,
                    target_image_path=target_image_path,
                    target_json_path=target_json_path,
                    target_total_path=target_total_path,
                    aggregate_document=aggregate_document,
                )
                processed_batches += 1

                print(
                    f"[BATCH PRELABEL] stem={stem} batch_start={batch_start} "
                    f"candidates={[item['id'] for item in batch_items]} "
                    f"finish_reason={finish_reason or '-'} "
                f"tokens={usage.get('total_tokens', '-') if isinstance(usage, dict) else '-'} "
                f"max_tokens={used_max_tokens} "
                f"reasoning={'on' if args.enable_reasoning else 'off'} "
                f"applied={applied_count} "
                f"kept_elements={label_ready_document['element_count']}"
            )

                if args.sleep_seconds > 0 and local_batch_index + 1 < max_batches:
                    time.sleep(args.sleep_seconds)
        except KeyboardInterrupt:
            interrupted = True
            label_ready_document = write_prelabel_outputs(
                source_image_path=image_path,
                target_image_path=target_image_path,
                target_json_path=target_json_path,
                target_total_path=target_total_path,
                aggregate_document=aggregate_document,
            )
            print()
            print(f"[INTERRUPTED] saved current progress for stem={stem}")
            print(f"              processed_batches={processed_batches}")
            print(f"              kept_elements={label_ready_document['element_count']}")

    status = "INTERRUPTED" if interrupted else "DONE"
    print(f"[{status}] source_json={json_path}")
    print(f"       source_image={image_path}")
    print(f"       target_image={target_image_path}")
    print(f"       target_json={target_json_path}")
    print(f"       target_total={target_total_path}")
    print(f"       processed_batches={processed_batches}")


if __name__ == "__main__":
    main()
