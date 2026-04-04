from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSING_ROOT_DIR = ROOT_DIR / "processing"
PRELABEL_DIR = PROCESSING_ROOT_DIR / "2.prelabel"
PRELABEL_IMAGE_DIR = PRELABEL_DIR / "ima"
PRELABEL_DATA_DIR = PRELABEL_DIR / "data"
PRELABEL_TOTAL_DIR = PRELABEL_DIR / "total"
FINAL_UNLABEL_DIR = PROCESSING_ROOT_DIR / "1.final_unlabel"
FINAL_UNLABEL_IMAGE_DIR = FINAL_UNLABEL_DIR / "ima"
FINAL_UNLABEL_TOTAL_DIR = FINAL_UNLABEL_DIR / "total"
PROCESS_IMAGE_DIR = ROOT_DIR / "process" / "ima"
PROCESS_TOTAL_DIR = ROOT_DIR / "process" / "total"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a root JSON file from processing/2.prelabel, convert valid prelabel results into the label-tool "
            "format, and write JSON/image/preview outputs into processing/2.prelabel/data, ima, and total."
        )
    )
    parser.add_argument(
        "--prelabel-dir",
        type=Path,
        default=PRELABEL_DIR,
        help="Directory containing prelabel JSON files.",
    )
    parser.add_argument(
        "--target-image-dir",
        type=Path,
        default=PRELABEL_IMAGE_DIR,
        help="Directory to write the preview image file used by the label tool.",
    )
    parser.add_argument(
        "--target-data-dir",
        type=Path,
        default=PRELABEL_DATA_DIR,
        help="Directory to write the converted JSON file.",
    )
    parser.add_argument(
        "--target-total-dir",
        type=Path,
        default=PRELABEL_TOTAL_DIR,
        help="Directory to write the current preview image copy.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="",
        help="Optional stem to convert instead of the first file in prelabel/.",
    )
    return parser.parse_args()


def ensure_dirs(image_dir: Path, data_dir: Path, total_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    total_dir.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def pick_prelabel_file(prelabel_dir: Path, stem: str) -> Path:
    if stem:
        candidate = prelabel_dir / f"{stem}_omniparser.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Prelabel file not found for stem: {stem}")
        return candidate

    files = sorted(prelabel_dir.glob("*_omniparser.json"))
    if not files:
        raise FileNotFoundError(f"No prelabel JSON files found in {prelabel_dir}")
    return files[0]


def extract_stem(json_path: Path) -> str:
    return json_path.stem.removesuffix("_omniparser")


def clamp_bbox(bbox: list[Any], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def bbox_center(bbox: list[int]) -> list[int]:
    return [int(round((bbox[0] + bbox[2]) / 2)), int(round((bbox[1] + bbox[3]) / 2))]


def ratio_bbox_to_pixels(bbox_ratio: list[Any], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in bbox_ratio[:4]]
    return clamp_bbox(
        [
            int(round(x1 * width)),
            int(round(y1 * height)),
            int(round(x2 * width)),
            int(round(y2 * height)),
        ],
        width,
        height,
    )


def bbox_ratio_from_pixels(bbox: list[int], width: int, height: int) -> list[float]:
    return [
        round(bbox[0] / width, 6),
        round(bbox[1] / height, 6),
        round(bbox[2] / width, 6),
        round(bbox[3] / height, 6),
    ]


def resolve_source_image(document: dict[str, Any], stem: str) -> tuple[Path, str]:
    image_value = str(document.get("image", "")).strip()
    candidates: list[tuple[Path, str]] = []

    if image_value:
        candidates.append((Path(image_value), "document_image"))

    candidates.extend(
        [
            (FINAL_UNLABEL_IMAGE_DIR / f"{stem}.png", "final_unlabel_image"),
            (PROCESS_IMAGE_DIR / f"{stem}.png", "process_image"),
            (FINAL_UNLABEL_TOTAL_DIR / f"{stem}_overlay.png", "final_unlabel_overlay"),
            (PROCESS_TOTAL_DIR / f"{stem}_overlay.png", "process_overlay"),
        ]
    )

    for path, source_kind in candidates:
        if path.exists():
            return path, source_kind

    raise FileNotFoundError(
        f"No source image found for stem {stem}. Checked image field, final_unlabel/ima, process/ima, and overlay paths."
    )


def convert_element(element: dict[str, Any], image_size: list[Any] | tuple[Any, Any]) -> dict[str, Any] | None:
    prelabel = element.get("prelabel", {})
    if not isinstance(prelabel, dict):
        return None
    if str(prelabel.get("validity", "")).strip().lower() != "valid":
        return None

    if len(image_size) != 2:
        width, height = 1, 1
    else:
        width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))

    bbox = element.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        pixel_bbox = clamp_bbox(bbox, width, height)
    else:
        bbox_ratio = element.get("bbox_ratio", [0, 0, 1, 1])
        pixel_bbox = ratio_bbox_to_pixels(bbox_ratio, width, height)

    name = str(prelabel.get("name", "")).strip() or str(element.get("name", "")).strip()
    element_type = str(prelabel.get("type", "")).strip() or str(element.get("type", "")).strip() or "icon_button"

    return {
        "id": str(element.get("id", "")),
        "name": name,
        "bbox": pixel_bbox,
        "bbox_ratio": bbox_ratio_from_pixels(pixel_bbox, width, height),
        "center": bbox_center(pixel_bbox),
        "raw_type": str(element.get("raw_type", "")).strip(),
        "clickable": bool(element.get("clickable", False)),
        "confidence": float(element.get("confidence", 0.0)),
        "region": str(element.get("region", "")).strip(),
        "source": str(element.get("source", "")).strip() or "prelabel_read",
        "type": element_type,
    }


def convert_document(document: dict[str, Any], target_image_path: Path) -> dict[str, Any]:
    image_size = document.get("image_size", [1, 1])
    converted_elements = [
        converted
        for converted in (convert_element(element, image_size) for element in document.get("elements", []))
        if converted is not None
    ]

    return {
        "image": str(target_image_path),
        "image_size": image_size,
        "raw_element_count": int(document.get("raw_element_count", len(document.get("elements", [])))),
        "element_count": len(converted_elements),
        "elements": converted_elements,
    }


def render_total_preview(source_image_path: Path, converted_document: dict[str, Any], target_total_path: Path) -> None:
    with Image.open(source_image_path) as image:
        canvas = image.convert("RGBA")

    draw = ImageDraw.Draw(canvas, "RGBA")
    outline_width = max(2, int(round(max(canvas.size) * 0.003)))

    for element in converted_document.get("elements", []):
        bbox = element.get("bbox", [0, 0, 1, 1])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in bbox]
        element_type = str(element.get("type", "")).strip()
        color = box_color_for_type(element_type)

        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=outline_width)

    target_total_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(target_total_path)


def main() -> None:
    args = parse_args()
    ensure_dirs(args.target_image_dir, args.target_data_dir, args.target_total_dir)

    prelabel_path = pick_prelabel_file(args.prelabel_dir, args.stem)
    stem = extract_stem(prelabel_path)
    document = load_json(prelabel_path)

    source_image_path, source_kind = resolve_source_image(document, stem)
    target_image_path = args.target_image_dir / f"{stem}.png"
    shutil.copy2(source_image_path, target_image_path)
    target_total_path = args.target_total_dir / f"{stem}_overlay.png"

    converted_document = convert_document(document, target_image_path)
    target_json_path = args.target_data_dir / prelabel_path.name
    write_json(target_json_path, converted_document)
    render_total_preview(source_image_path, converted_document, target_total_path)

    print(f"[DONE] prelabel={prelabel_path}")
    print(f"       image_source={source_image_path} ({source_kind})")
    print(f"       image_target={target_image_path}")
    print(f"       total_target={target_total_path}")
    print(f"       json_target={target_json_path}")
    print(f"       kept_elements={converted_document['element_count']}")


if __name__ == "__main__":
    main()
