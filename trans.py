from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an OmniParser-style JSON that contains prelabel results into a label-tool-readable JSON file."
        )
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to the source *_omniparser.json file, for example a batch_prelabel_test_output aggregate file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <input_dir>/<stem>_label_ready.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clamp_bbox(bbox: list[Any], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


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


def bbox_center(bbox: list[int]) -> list[int]:
    return [int(round((bbox[0] + bbox[2]) / 2)), int(round((bbox[1] + bbox[3]) / 2))]


def convert_element(element: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any] | None:
    prelabel = element.get("prelabel", {})
    if not isinstance(prelabel, dict):
        return None
    if str(prelabel.get("validity", "")).strip().lower() != "valid":
        return None

    width, height = image_size
    bbox = element.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        pixel_bbox = clamp_bbox(bbox, width, height)
    else:
        bbox_ratio = element.get("bbox_ratio")
        if not isinstance(bbox_ratio, list) or len(bbox_ratio) != 4:
            return None
        pixel_bbox = ratio_bbox_to_pixels(bbox_ratio, width, height)

    name = str(prelabel.get("name", "")).strip() or str(element.get("name", "")).strip()
    element_type = str(prelabel.get("type", "")).strip() or str(element.get("type", "")).strip() or "icon_button"
    clickable = bool(prelabel.get("clickable", element.get("clickable", False)))
    instruction = str(prelabel.get("instruction", "")).strip() or str(element.get("instruction", "")).strip()

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
        "source": str(element.get("source", "")).strip() or "trans",
        "type": element_type,
    }


def convert_document(document: dict[str, Any]) -> dict[str, Any]:
    image_size_raw = document.get("image_size", [1, 1])
    if not isinstance(image_size_raw, list) or len(image_size_raw) != 2:
        image_size_raw = [1, 1]
    width = max(1, int(image_size_raw[0]))
    height = max(1, int(image_size_raw[1]))
    image_size = (width, height)

    converted_elements = [
        converted
        for converted in (convert_element(element, image_size) for element in document.get("elements", []))
        if converted is not None
    ]

    return {
        "image": str(document.get("image", "")),
        "image_size": [width, height],
        "raw_element_count": int(document.get("raw_element_count", len(document.get("elements", [])))),
        "element_count": len(converted_elements),
        "elements": converted_elements,
    }


def default_output_path(input_json: Path) -> Path:
    stem = input_json.stem.removesuffix("_omniparser")
    return input_json.with_name(f"{stem}_label_ready.json")


def main() -> None:
    args = parse_args()
    input_json = args.input_json.resolve()
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    source_document = load_json(input_json)
    converted_document = convert_document(source_document)
    output_json = args.output_json.resolve() if args.output_json is not None else default_output_path(input_json)
    write_json(output_json, converted_document)

    print(f"[DONE] input={input_json}")
    print(f"       output={output_json}")
    print(f"       kept_elements={converted_document['element_count']}")


if __name__ == "__main__":
    main()
