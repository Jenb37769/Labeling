from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSING_ROOT_DIR = ROOT_DIR / "processing"
DEFAULT_IMAGE_DIR = PROCESSING_ROOT_DIR / "1.final_unlabel" / "ima"
DEFAULT_DATA_DIR = PROCESSING_ROOT_DIR / "1.final_unlabel" / "data"
DEFAULT_PRELABEL_DIR = PROCESSING_ROOT_DIR / "2.prelabel"
REQUEST_RETRY_COUNT = 3
REQUEST_RETRY_SLEEP_SECONDS = 2.0

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
    "You are helping prelabel Photoshop GUI elements from tutorial screenshots. "
    "You will be shown two images for one candidate element: "
    "1) the original screenshot with the target highlighted, "
    "2) a zoomed crop of the same target. "
    "Decide whether the highlighted target is a valid UI annotation target. "
    "If it is valid, choose the best type from the allowed type list and provide a concise English snake_case name. "
    "If it is not a valid annotation target, mark it invalid and leave type/name empty. "
    "Be conservative. Do not invent details you cannot see."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prelabel OmniParser elements with an OpenAI-compatible vision API. "
            "By default this script writes JSON copies into the prelabel directory without modifying the original "
            "final_unlabel JSON files."
        )
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory containing source PNGs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing *_omniparser.json files.",
    )
    parser.add_argument("--stem", type=str, default="", help="Only process one frame stem.")
    parser.add_argument("--limit-files", type=int, default=0, help="Process at most this many JSON files.")
    parser.add_argument(
        "--limit-elements",
        type=int,
        default=0,
        help="Process at most this many elements across all files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run elements that already have a prelabel block and overwrite suggested name/type.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write JSON files; only print what would be updated.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between requests to reduce provider throttling.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="HTTP timeout for each LLM request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Max completion tokens for the prelabel response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("PRELABEL_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")),
        help="Vision-capable model name.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("PRELABEL_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("PRELABEL_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        help="API key for the provider. Falls back to PRELABEL_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--prelabel-dir",
        type=Path,
        default=DEFAULT_PRELABEL_DIR,
        help="Directory to store output JSON copies for prelabel results.",
    )
    parser.add_argument(
        "--write-source",
        action="store_true",
        help="Also write results back into the original final_unlabel JSON files.",
    )
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Pass a reasoning block for providers such as OpenRouter that support it.",
    )
    parser.add_argument(
        "--http-referer",
        type=str,
        default=os.environ.get("PRELABEL_HTTP_REFERER", ""),
        help="Optional HTTP-Referer header for providers such as OpenRouter.",
    )
    parser.add_argument(
        "--app-title",
        type=str,
        default=os.environ.get("PRELABEL_APP_TITLE", "grounding-prelabel"),
        help="Optional X-Title header for providers such as OpenRouter.",
    )
    args = parser.parse_args()
    args.model = str(args.model or "").strip()
    args.base_url = str(args.base_url or "").strip()
    args.api_key = str(args.api_key or "").strip()
    args.http_referer = str(args.http_referer or "").strip()
    args.app_title = str(args.app_title or "").strip()
    args.prelabel_dir = Path(args.prelabel_dir)
    return args


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp_bbox(bbox: list[Any], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def expand_bbox(
    bbox: list[int],
    width: int,
    height: int,
    padding_ratio: float = 0.35,
    min_padding: int = 20,
) -> list[int]:
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


def build_context_image(image: Image.Image, bbox: list[int]) -> Image.Image:
    base = image.convert("RGBA")
    shaded = Image.new("RGBA", base.size, (0, 0, 0, 115))
    dimmed = Image.alpha_composite(base, shaded)

    x1, y1, x2, y2 = bbox
    target_region = base.crop((x1, y1, x2, y2))
    dimmed.paste(target_region, (x1, y1))

    draw = ImageDraw.Draw(dimmed)
    outline_width = max(2, int(round(max(base.size) * 0.003)))
    draw.rectangle((x1, y1, x2, y2), outline=(255, 59, 48, 255), width=outline_width)
    return dimmed.convert("RGB")


def resize_image_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    longest_side = max(image.size)
    if longest_side <= max_side:
        return image
    scale = max_side / max(longest_side, 1)
    return image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )


def build_crop_image(image: Image.Image, bbox: list[int], max_side: int = 768) -> Image.Image:
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
    draw.rectangle(tuple(local_bbox), outline=(255, 59, 48), width=outline_width)

    longest_side = max(crop.size)
    if longest_side >= max_side:
        return crop

    scale = max_side / max(longest_side, 1)
    return crop.resize(
        (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
        Image.Resampling.LANCZOS,
    )


def image_to_data_url(image: Image.Image, *, format: str = "JPEG", quality: int = 75) -> str:
    buffer = io.BytesIO()
    if format.upper() == "JPEG":
        image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        mime_type = "image/jpeg"
    else:
        image.save(buffer, format="PNG")
        mime_type = "image/png"
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def iter_json_paths(data_dir: Path, stem: str, limit_files: int) -> list[Path]:
    if stem:
        path = data_dir / f"{stem}_omniparser.json"
        return [path] if path.exists() else []

    paths = sorted(data_dir.glob("*_omniparser.json"))
    if limit_files > 0:
        return paths[:limit_files]
    return paths


def extract_stem(json_path: Path) -> str:
    return json_path.stem.removesuffix("_omniparser")


def resolve_image_path(image_dir: Path, json_path: Path, document: dict[str, Any]) -> Path:
    image_name = Path(str(document.get("image", ""))).name
    if image_name:
        direct = image_dir / image_name
        if direct.exists():
            return direct
    return image_dir / f"{extract_stem(json_path)}.png"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_prelabel_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_prelabel_copy(output_dir: Path, json_name: str, document: dict[str, Any]) -> None:
    ensure_prelabel_dir(output_dir)
    write_json(output_dir / json_name, document)


def normalize_type(value: Any) -> str:
    normalized = "_".join(str(value or "").strip().lower().replace("-", "_").split())
    return normalized if normalized in ALLOWED_TYPES else ""


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


def compute_bbox_ratios(bbox: list[int], image_size: list[Any] | tuple[Any, Any]) -> tuple[float, float, float, float]:
    if len(image_size) != 2:
        return 0.0, 0.0, 0.0, 0.0
    width = max(1, int(image_size[0]))
    height = max(1, int(image_size[1]))
    x1, y1, x2, y2 = bbox
    return (
        x1 / width,
        y1 / height,
        max(1, x2 - x1) / width,
        max(1, y2 - y1) / height,
    )


def axis_bucket(value: float, low_label: str, mid_label: str, high_label: str) -> str:
    if value < 0.33:
        return low_label
    if value > 0.67:
        return high_label
    return mid_label


def describe_position_metadata(element: dict[str, Any], image_size: list[Any] | tuple[Any, Any]) -> dict[str, Any]:
    bbox = element.get("bbox", [0, 0, 1, 1])
    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = [0, 0, 1, 1]

    if len(image_size) != 2:
        image_size = [1, 1]
    width = max(1, int(image_size[0]))
    height = max(1, int(image_size[1]))

    center = element.get("center")
    if not isinstance(center, list) or len(center) != 2:
        center = [int(round((bbox[0] + bbox[2]) / 2)), int(round((bbox[1] + bbox[3]) / 2))]

    center_x_ratio = center[0] / width
    center_y_ratio = center[1] / height
    x_ratio, y_ratio, w_ratio, h_ratio = compute_bbox_ratios(bbox, image_size)

    horizontal = axis_bucket(center_x_ratio, "left", "center", "right")
    vertical = axis_bucket(center_y_ratio, "top", "middle", "bottom")
    region = str(element.get("region", "")).strip()
    region_meaning_map = {
        "top_bar": "top application or menu bar area",
        "left_toolbar": "left toolbar area",
        "right_panel": "right-side panel area",
        "canvas_overlay": "main canvas area",
        "other": "other or non-core area",
    }
    region_meaning = region_meaning_map.get(region, "other or non-core area")

    if region in {"top_bar", "left_toolbar", "right_panel"}:
        screening_hint = (
            "This position is in a core Photoshop UI area, so a valid control, icon, tab, menu item, or nearby explanatory label is more likely."
        )
    elif region == "canvas_overlay":
        screening_hint = (
            "This position is in the main canvas area. Treat it as valid only if it is clearly an interactive control, canvas target, or a local explanatory label tied to nearby controls."
        )
    else:
        screening_hint = (
            "This position is outside the main Photoshop UI zones. Be conservative and mark invalid unless the target is clearly a useful UI control or a direct explanation of nearby controls."
        )

    return {
        "center": center,
        "center_ratio": [round(center_x_ratio, 4), round(center_y_ratio, 4)],
        "bbox_ratio": [round(x_ratio, 4), round(y_ratio, 4), round(w_ratio, 4), round(h_ratio, 4)],
        "position_bucket": f"{vertical}_{horizontal}",
        "region": region,
        "region_meaning": region_meaning,
        "screening_hint": screening_hint,
    }


def shared_rule_lines(*, include_clickable_instruction: bool) -> list[str]:
    lines = [
        " - A target can be valid only if it corresponds to exactly one complete, atomic Photoshop UI element.",
        " - Atomic means a single, independently meaningful UI target, such as one icon, one button, one menu item, one tab, one slider, one dropdown, one input field, or one labeled standalone control.",
       "  - If the highlighted crop shows two or more recognizable icons/controls, validity=invalid, even if one icon is centered, larger, or more visually salient.",
       "  - Do not guess the intended target based on crop center, position, or the label Current Icon. Judge only whether the highlighted content itself is exactly one complete target.",
       "  - Invalid conditions have priority over valid interpretation. If any invalid rule applies, output invalid.",
       "  - First decide the target count: zero, one, or multiple complete UI elements. Only a count of exactly one may be valid.",
       "  - If the crop contains only a fragment of an icon/control, validity=invalid, even if the missing parts are easy to infer.",
       "  - If the crop contains separator lines, neighboring controls, or toolbar groups together rather than one isolated target, validity=invalid.",
        "- type must be empty when validity is invalid or uncertain.",
        "- name must be empty when validity is invalid or uncertain.",
    ]
    if include_clickable_instruction:
        lines.extend(
            [
                "- clickable must be false when validity is invalid or uncertain.",
                "- instruction must be empty when validity is invalid or uncertain.",
            ]
        )
    lines.extend(
        [
            "- name must be concise English snake_case.",
            "- Prefer stable Photoshop UI names such as brush_tool, layers_tab, file_menu, opacity_slider.",
            "- Use both visual evidence and position metadata as an early screening signal.",
            "- Core Photoshop UI zones are more likely to contain valid controls. Non-core positions should be treated conservatively.",
            "- Text can still be valid if it clearly explains nearby icons or controls.",
        ]
    )
    if include_clickable_instruction:
        lines.append(
            "- instruction must be a short English action or function description such as select_the_brush_tool, toggle_layer_visibility, or open_alignment_options."
        )
    lines.append("- Do not output any extra markdown, code fences, or explanation outside the JSON.")
    return lines


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = strip_code_fences(text)
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON: {text[:200]}")
    return json.loads(stripped[start : end + 1])


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
                elif "text" in item:
                    texts.append(str(item["text"]))
            else:
                texts.append(str(item))
        return "\n".join(part for part in texts if part)
    return str(content)


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 10:
        return "*" * len(api_key)
    return f"{api_key[:6]}...{api_key[-4:]}"


def likely_provider_hint(base_url: str, model: str, api_key: str) -> str:
    lowered_base = base_url.lower()
    lowered_model = model.lower()

    if api_key.startswith("sk-or-v1") and "api.openai.com" in lowered_base:
        return (
            "This key looks like an OpenRouter key, but the base URL points to OpenAI. "
            "Try --base-url https://openrouter.ai/api/v1."
        )
    if "openrouter.ai" in lowered_base:
        return (
            "This request is targeting OpenRouter. If the model is not available on your account, "
            "try a different routed model or confirm the exact model slug in OpenRouter."
        )
    if "/" in lowered_model and "api.openai.com" in lowered_base:
        return (
            "This model name looks provider-routed rather than an OpenAI native model. "
            "If you intended to use OpenRouter, set --base-url https://openrouter.ai/api/v1."
        )
    return ""


def call_openai_compatible_api(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
    max_tokens: int,
    temperature: float,
    prompt_text: str,
    context_data_url: str,
    crop_data_url: str,
    enable_reasoning: bool,
    http_referer: str,
    app_title: str,
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    if not api_key:
        raise ValueError("Missing API key. Set PRELABEL_API_KEY or pass --api-key.")

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": context_data_url}},
                    {"type": "image_url", "image_url": {"url": crop_data_url}},
                ],
            },
        ],
    }
    if enable_reasoning:
        payload["reasoning"] = {"enabled": True}

    request_body = json.dumps(payload).encode("utf-8")
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "grounding-prelabel/1.0",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if app_title:
        headers["X-Title"] = app_title

    session = requests.Session()
    session.trust_env = False

    last_error: Exception | None = None
    response_text = ""
    for attempt in range(1, REQUEST_RETRY_COUNT + 1):
        try:
            response = session.post(
                endpoint,
                data=request_body,
                headers=headers,
                timeout=timeout_seconds,
            )
            response_text = response.text
            response.raise_for_status()
            break
        except requests.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else 0
            if status_code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= REQUEST_RETRY_COUNT:
                detail = exc.response.text if exc.response is not None else ""
                hint = likely_provider_hint(base_url, model, api_key)
                masked = mask_api_key(api_key)
                message = (
                    f"API request failed with HTTP {status_code or 'unknown'}.\n"
                    f"base_url={base_url}\n"
                    f"model={model}\n"
                    f"api_key={masked}\n"
                    f"provider_hint={hint or '(none)'}\n"
                    f"detail={detail}"
                )
                raise RuntimeError(message) from exc
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= REQUEST_RETRY_COUNT:
                raise RuntimeError(f"API request failed after {REQUEST_RETRY_COUNT} attempts: {exc}") from exc

        print(
            f"[RETRY] request failed for model={model} attempt={attempt}/{REQUEST_RETRY_COUNT}; "
            f"retrying in {REQUEST_RETRY_SLEEP_SECONDS:.1f}s"
        )
        time.sleep(REQUEST_RETRY_SLEEP_SECONDS)
    else:
        raise RuntimeError(f"API request failed after {REQUEST_RETRY_COUNT} attempts: {last_error}")

    response_json = json.loads(response_text)
    try:
        message_content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {response_text[:500]}") from exc

    raw_text = flatten_message_content(message_content)
    parsed = extract_json_object(raw_text)
    finish_reason = str(response_json.get("choices", [{}])[0].get("finish_reason", ""))
    usage = response_json.get("usage", {})
    return parsed, raw_text, finish_reason, usage


def build_prompt_text(element: dict[str, Any], stem: str, image_size: list[Any] | tuple[Any, Any]) -> str:
    bbox = element.get("bbox", [0, 0, 0, 0])
    region = element.get("region", "")
    raw_type = element.get("raw_type", "")
    clickable = bool(element.get("clickable", False))
    existing_name = str(element.get("name", "")).strip()
    position_info = describe_position_metadata(element, image_size)

    return (
        "You are a professional labeling assistant. Output exactly one complete JSON object that must strictly follow the schema below. "
        "Do not output anything extra, and do not omit any part of the structured output.\n"
        "{\n"
        '  "validity": "valid" | "invalid" | "uncertain",\n'
        f'  "type": one of {json.dumps(ALLOWED_TYPES)},\n'
        '  "name": "english_snake_case_or_empty",\n'
        '  "reason": "short explanation",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "Rules:\n"
        + "\n".join(shared_rule_lines(include_clickable_instruction=False))
        + "\n\n"
        "Candidate metadata:\n"
        f"- frame_stem: {stem}\n"
        f"- element_id: {element.get('id', '')}\n"
        f"- image_size: {list(image_size)}\n"
        f"- bbox: {bbox}\n"
        f"- center: {position_info['center']}\n"
        f"- bbox_ratio_xywh: {position_info['bbox_ratio']}\n"
        f"- center_ratio_xy: {position_info['center_ratio']}\n"
        f"- position_bucket: {position_info['position_bucket']}\n"
        f"- region: {region}\n"
        f"- region_meaning: {position_info['region_meaning']}\n"
        f"- position_screening_hint: {position_info['screening_hint']}\n"
        f"- raw_type_from_omniparser: {raw_type}\n"
        f"- clickable_from_omniparser: {str(clickable).lower()}\n"
        f"- existing_name_from_omniparser: {existing_name or '(empty)'}\n"
        "- The first image is the full screenshot with only the target highlighted.\n"
        "- The second image is a zoomed crop of the same target.\n"
    )


def prelabel_result_from_model_response(
    response_data: dict[str, Any],
    raw_response_text: str,
    *,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    validity = str(response_data.get("validity", "")).strip().lower()
    if validity not in {"valid", "invalid", "uncertain"}:
        validity = "uncertain"

    normalized_type = normalize_type(response_data.get("type", ""))
    normalized_name = normalize_name(response_data.get("name", ""))

    if validity != "valid":
        normalized_type = ""
        normalized_name = ""

    try:
        confidence = float(response_data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "validity": validity,
        "type": normalized_type,
        "name": normalized_name,
        "reason": str(response_data.get("reason", "")).strip(),
        "confidence": round(confidence, 4),
        "model": model,
        "base_url": base_url,
        "raw_response": raw_response_text,
        "updated_at": utc_now_iso(),
    }


def apply_prelabel_to_element(
    element: dict[str, Any],
    prelabel_result: dict[str, Any],
    *,
    overwrite: bool,
) -> bool:
    previous_name = str(element.get("name", ""))
    previous_type = str(element.get("type", ""))

    element["prelabel"] = {
        **prelabel_result,
        "previous_name": previous_name,
        "previous_type": previous_type,
    }

    changed = True
    if prelabel_result["validity"] == "valid":
        suggested_name = prelabel_result["name"]
        suggested_type = prelabel_result["type"]

        if overwrite or not str(element.get("name", "")).strip():
            if suggested_name:
                element["name"] = suggested_name
        if suggested_type:
            element["type"] = suggested_type

    return changed


def should_skip_element(element: dict[str, Any], overwrite: bool) -> bool:
    if overwrite:
        return False
    prelabel = element.get("prelabel")
    return isinstance(prelabel, dict) and bool(prelabel.get("updated_at"))


def process_json_file(json_path: Path, args: argparse.Namespace, remaining_budget: int | None) -> tuple[int, int]:
    document = load_json(json_path)
    image_path = resolve_image_path(args.image_dir, json_path, document)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for {json_path.name}: {image_path}")

    stem = extract_stem(json_path)
    processed_count = 0
    changed_count = 0

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        elements = document.get("elements", [])
        for element in elements:
            if remaining_budget is not None and processed_count >= remaining_budget:
                break
            if should_skip_element(element, args.overwrite):
                continue

            bbox = element.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            clamped_bbox = clamp_bbox(bbox, image.width, image.height)
            context_image = resize_image_to_max_side(build_context_image(image, clamped_bbox), max_side=1280)
            crop_image = build_crop_image(image, clamped_bbox)
            element_id = str(element.get("id", ""))

            prompt_text = build_prompt_text(element, stem, document.get("image_size", [image.width, image.height]))
            response_data, raw_response_text, finish_reason, usage = call_openai_compatible_api(
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                prompt_text=prompt_text,
                context_data_url=image_to_data_url(context_image, format="JPEG", quality=72),
                crop_data_url=image_to_data_url(crop_image, format="JPEG", quality=80),
                enable_reasoning=args.enable_reasoning,
                http_referer=args.http_referer,
                app_title=args.app_title,
            )
            prelabel_result = prelabel_result_from_model_response(
                response_data,
                raw_response_text,
                model=args.model,
                base_url=args.base_url,
            )
            prelabel_result["finish_reason"] = finish_reason
            prelabel_result["usage"] = usage
            apply_prelabel_to_element(element, prelabel_result, overwrite=args.overwrite)
            processed_count += 1
            changed_count += 1
            document["prelabel_meta"] = {
                "model": args.model,
                "base_url": args.base_url,
                "updated_at": utc_now_iso(),
            }
            if not args.dry_run:
                write_prelabel_copy(args.prelabel_dir, json_path.name, document)

            print(
                f"[PRELABEL] {json_path.name} element={element.get('id', '')} "
                f"validity={prelabel_result['validity']} "
                f"type={prelabel_result['type'] or '-'} "
                f"name={prelabel_result['name'] or '-'} "
                f"confidence={prelabel_result['confidence']:.2f} "
                f"finish_reason={finish_reason or '-'} "
                f"tokens={usage.get('total_tokens', '-') if isinstance(usage, dict) else '-'}"
            )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    if changed_count and not args.dry_run:
        if args.write_source:
            write_json(json_path, document)

    return processed_count, changed_count


def main() -> None:
    args = parse_args()
    json_paths = iter_json_paths(args.data_dir, args.stem, args.limit_files)
    if not json_paths:
        print(f"No OmniParser JSON files found in {args.data_dir}")
        return

    total_processed = 0
    total_changed = 0
    limit_remaining: int | None = args.limit_elements if args.limit_elements > 0 else None

    for json_path in json_paths:
        if limit_remaining is not None and limit_remaining <= 0:
            break

        processed_count, changed_count = process_json_file(json_path, args, limit_remaining)
        total_processed += processed_count
        total_changed += changed_count
        if limit_remaining is not None:
            limit_remaining -= processed_count

    mode = "DRY RUN" if args.dry_run else "DONE"
    print()
    print(
        f"[{mode}] files={len(json_paths)} processed_elements={total_processed} "
        f"updated_elements={total_changed} model={args.model}"
    )


if __name__ == "__main__":
    main()
