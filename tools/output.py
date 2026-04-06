from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT_DIR / "script"
PROCESSING_ROOT_DIR = ROOT_DIR / "processing"
SOURCE_IMAGE_DIR = PROCESSING_ROOT_DIR / "1.final_unlabel" / "ima"
SOURCE_DATA_DIR = PROCESSING_ROOT_DIR / "1.final_unlabel" / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prelabel as single_prelabel  # noqa: E402
import test_prelabel_batch as batch_prelabel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the current LLM inputs used by script/prelabel.py and script/test_prelabel_batch.py "
            "for the first complete sample in processing/1.final_unlabel."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=SOURCE_DATA_DIR)
    parser.add_argument("--image-dir", type=Path, default=SOURCE_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--icon-per-batch", type=int, default=4)
    return parser.parse_args()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def find_first_complete_sample(data_dir: Path, image_dir: Path) -> tuple[Path, Path]:
    for json_path in sorted(data_dir.glob("*_omniparser.json")):
        stem = json_path.stem.removesuffix("_omniparser")
        image_path = image_dir / f"{stem}.png"
        if image_path.exists():
            return json_path, image_path
    raise FileNotFoundError(
        f"No complete sample found in {data_dir} and {image_dir}. A complete sample needs both JSON and PNG."
    )


def find_first_bbox_element(document: dict[str, Any]) -> dict[str, Any]:
    for element in document.get("elements", []):
        bbox = element.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            return element
    raise ValueError("No element with a valid bbox was found in the source JSON.")


def export_batch_input(
    *,
    image,
    document: dict[str, Any],
    stem: str,
    output_dir: Path,
    icon_per_batch: int,
) -> None:
    batch_dir = output_dir / "test_prelabel_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    candidates = batch_prelabel.collect_candidates(document, image.size)
    batch_items = candidates[: max(1, icon_per_batch)]
    if not batch_items:
        raise ValueError("No batch candidates found.")

    overview_image = batch_prelabel.resize_image_to_max_side(
        batch_prelabel.build_overview_image(image, batch_items),
        max_side=1600,
    )
    overview_path = batch_dir / "overview.jpg"
    overview_image.save(overview_path, format="JPEG", quality=82, optimize=True)

    crop_paths: list[str] = []
    for item in batch_items:
        crop_image = batch_prelabel.resize_image_to_max_side(
            batch_prelabel.build_crop_image(image, item["bbox"], item["id"]),
            max_side=768,
        )
        crop_path = batch_dir / f"crop_{item['id']}.jpg"
        crop_image.save(crop_path, format="JPEG", quality=84, optimize=True)
        crop_paths.append(str(crop_path))

    prompt_text = batch_prelabel.build_prompt_text(stem, image.size, batch_items)
    write_text(batch_dir / "prompt.txt", prompt_text)
    write_json(
        batch_dir / "meta.json",
        {
            "stem": stem,
            "candidate_ids": [item["id"] for item in batch_items],
            "image_count_sent_to_llm": 1 + len(batch_items),
            "overview_image": str(overview_path),
            "crop_images": crop_paths,
        },
    )


def export_single_prelabel_input(
    *,
    image,
    document: dict[str, Any],
    stem: str,
    output_dir: Path,
) -> None:
    prelabel_dir = output_dir / "prelabel"
    prelabel_dir.mkdir(parents=True, exist_ok=True)

    element = find_first_bbox_element(document)
    bbox = single_prelabel.clamp_bbox(element["bbox"], image.width, image.height)
    context_image = single_prelabel.resize_image_to_max_side(
        single_prelabel.build_context_image(image, bbox),
        max_side=1280,
    )
    crop_image = single_prelabel.build_crop_image(image, bbox)

    context_path = prelabel_dir / "context.jpg"
    crop_path = prelabel_dir / "crop.jpg"
    context_image.save(context_path, format="JPEG", quality=82, optimize=True)
    crop_image.save(crop_path, format="JPEG", quality=84, optimize=True)

    prompt_text = single_prelabel.build_prompt_text(element, stem, document.get("image_size", [image.width, image.height]))
    write_text(prelabel_dir / "prompt.txt", prompt_text)
    write_json(
        prelabel_dir / "meta.json",
        {
            "stem": stem,
            "element_id": str(element.get("id", "")),
            "bbox": bbox,
            "image_count_sent_to_llm": 2,
            "context_image": str(context_path),
            "crop_image": str(crop_path),
        },
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    json_path, image_path = find_first_complete_sample(args.data_dir, args.image_dir)
    stem = json_path.stem.removesuffix("_omniparser")
    document = batch_prelabel.load_json(json_path)

    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        export_batch_input(
            image=image,
            document=document,
            stem=stem,
            output_dir=output_root,
            icon_per_batch=args.icon_per_batch,
        )
        export_single_prelabel_input(
            image=image,
            document=document,
            stem=stem,
            output_dir=output_root,
        )

    write_json(
        output_root / "source.json",
        {
            "source_json": str(json_path),
            "source_image": str(image_path),
            "stem": stem,
        },
    )

    print(f"[DONE] source_json={json_path}")
    print(f"       source_image={image_path}")
    print(f"       batch_output={output_root / 'test_prelabel_batch'}")
    print(f"       prelabel_output={output_root / 'prelabel'}")


if __name__ == "__main__":
    main()
