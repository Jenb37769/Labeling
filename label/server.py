from __future__ import annotations

import json
import mimetypes
import shutil
import sys
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from math import dist
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT_DIR / "label"
PROCESSING_ROOT_DIR = ROOT_DIR / "processing"
PRELABEL_IMAGE_DIR = PROCESSING_ROOT_DIR / "2.prelabel" / "ima"
PRELABEL_DATA_DIR = PROCESSING_ROOT_DIR / "2.prelabel" / "data"
PRELABEL_TOTAL_DIR = PROCESSING_ROOT_DIR / "2.prelabel" / "total"
FINAL_LABEL_IMAGE_DIR = PROCESSING_ROOT_DIR / "3.final_label" / "ima"
FINAL_LABEL_DATA_DIR = PROCESSING_ROOT_DIR / "3.final_label" / "data"
FINAL_LABEL_TOTAL_DIR = PROCESSING_ROOT_DIR / "3.final_label" / "total"
HIST_PATH = ROOT_DIR / "hist.json"
LOAD_COUNT_PATH = ROOT_DIR / "load_counts.json"
HOST = "127.0.0.1"
PORT = 8765


def ensure_dirs() -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_LABEL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_LABEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_LABEL_TOTAL_DIR.mkdir(parents=True, exist_ok=True)


def image_stems() -> list[str]:
    stems = []
    for image_path in sorted(PRELABEL_IMAGE_DIR.glob("*.png")):
        if image_path.name.startswith("."):
            continue
        stems.append(image_path.stem)
    return stems


def unlabeled_json_path(stem: str) -> Path:
    return PRELABEL_DATA_DIR / f"{stem}_omniparser.json"


def labeled_json_path(stem: str) -> Path:
    return FINAL_LABEL_DATA_DIR / f"{stem}.json"


def labeled_image_path(stem: str) -> Path:
    return FINAL_LABEL_IMAGE_DIR / f"{stem}.png"


def unlabeled_image_path(stem: str) -> Path:
    return PRELABEL_IMAGE_DIR / f"{stem}.png"


def unlabeled_diff_path(stem: str) -> Path:
    return PRELABEL_DATA_DIR / f"{stem}_diff.json"


def unlabeled_total_path(stem: str) -> Path:
    return PRELABEL_TOTAL_DIR / f"{stem}_overlay.png"


def labeled_total_path(stem: str) -> Path:
    return FINAL_LABEL_TOTAL_DIR / f"{stem}_overlay.png"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_hist() -> dict:
    if not HIST_PATH.exists():
        return {"count": 0, "items": [], "images": []}
    return load_json(HIST_PATH)


def hist_top_items(limit: int = 1000) -> list[dict]:
    hist = load_hist()
    items = list(hist.get("items", []))
    items.sort(key=lambda item: int(item.get("count", 0)), reverse=True)
    return items[:limit]


def save_hist(hist: dict) -> None:
    items = hist.get("items", [])
    items.sort(key=lambda item: int(item.get("count", 0)), reverse=True)
    hist["items"] = items
    HIST_PATH.write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")


def load_load_counts() -> dict[str, int]:
    if not LOAD_COUNT_PATH.exists():
        return {}
    raw = load_json(LOAD_COUNT_PATH)
    return {str(key): int(value) for key, value in raw.items()}


def save_load_counts(counts: dict[str, int]) -> None:
    LOAD_COUNT_PATH.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")


def element_size(bbox: list[int]) -> tuple[int, int, int]:
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    area = width * height
    return width, height, area


def is_close_ratio(reference: int, current: int, tolerance: float) -> bool:
    if reference <= 0:
        return False
    return abs(current - reference) / reference <= tolerance


def center_distance_ratio(center_a: list[int], center_b: list[int], baseline: float) -> float:
    if len(center_a) != 2 or len(center_b) != 2:
        return 1.0
    if baseline <= 0:
        return 1.0
    return dist(center_a, center_b) / baseline


def find_hist_match(
    items: list[dict],
    element: dict,
    image_size: tuple[int, int],
    tolerance: float = 0.10,
    center_tolerance: float = 0.08,
) -> dict | None:
    bbox = element.get("bbox", [0, 0, 1, 1])
    width, height, area = element_size(bbox)
    element_type = element.get("type", "")
    element_name = str(element.get("name", "")).strip()
    element_center = element.get("center", [])
    element_center_baseline = min(width, height)
    element_raw_type = str(element.get("raw_type", "")).strip()
    element_region = str(element.get("region", "")).strip()
    element_clickable = bool(element.get("clickable", False))

    for item in items:
        if item.get("type") != element_type:
            continue
        if str(item.get("name", "")).strip() != element_name:
            continue
        if str(item.get("raw_type", "")).strip() != element_raw_type:
            continue
        if str(item.get("region", "")).strip() != element_region:
            continue
        if bool(item.get("clickable", False)) != element_clickable:
            continue
        if center_distance_ratio(item.get("center", []), element_center, element_center_baseline) > center_tolerance:
            continue
        item_width = int(item.get("width", 0))
        item_height = int(item.get("height", 0))
        item_area = int(item.get("area", 0))
        if (
            is_close_ratio(item_width, width, tolerance)
            and is_close_ratio(item_height, height, tolerance)
            and is_close_ratio(item_area, area, tolerance)
        ):
            return item
    return None


def update_hist_on_save(stem: str, elements: list[dict], image_size: tuple[int, int]) -> None:
    hist = load_hist()
    items = list(hist.get("items", []))
    images = list(hist.get("images", []))

    for element in elements:
        match = find_hist_match(items, element, image_size)
        if match is not None:
            match["count"] = int(match.get("count", 0)) + 1
            continue

        bbox = element.get("bbox", [0, 0, 1, 1])
        width, height, area = element_size(bbox)
        items.append(
            {
                "type": element.get("type", ""),
                "name": element.get("name", ""),
                "raw_type": element.get("raw_type", ""),
                "region": element.get("region", ""),
                "clickable": bool(element.get("clickable", False)),
                "bbox": bbox,
                "center": element.get("center", []),
                "width": width,
                "height": height,
                "area": area,
                "count": 1,
            }
        )

    images.append(
        {
            "image": f"{stem}.png",
            "element_count": len(elements),
            "saved_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )

    hist["count"] = int(hist.get("count", 0)) + 1
    hist["items"] = items
    hist["images"] = images
    save_hist(hist)


def default_document(stem: str) -> dict:
    return {
        "image": f"{stem}.png",
        "elements": [],
    }


def normalize_element(element: dict, fallback_id: int) -> dict:
    bbox = [int(round(value)) for value in element.get("bbox", [0, 0, 1, 1])]
    if len(bbox) != 4:
        bbox = [0, 0, 1, 1]
    bbox[2] = max(bbox[0] + 1, bbox[2])
    bbox[3] = max(bbox[1] + 1, bbox[3])
    center = [int(round((bbox[0] + bbox[2]) / 2)), int(round((bbox[1] + bbox[3]) / 2))]
    return {
        "id": str(element.get("id", fallback_id)),
        "name": str(element.get("name", "")).strip(),
        "bbox": bbox,
        "center": center,
        "type": str(element.get("type", "icon_button")).strip() or "icon_button",
        "clickable": bool(element.get("clickable", True)),
        "raw_type": str(element.get("raw_type", "")).strip(),
        "region": str(element.get("region", "")).strip(),
    }


def load_document(stem: str) -> tuple[dict, str]:
    labeled_path = labeled_json_path(stem)
    if labeled_path.exists():
        document = load_json(labeled_path)
        source = "final_label"
    else:
        source_path = unlabeled_json_path(stem)
        if source_path.exists():
            source_doc = load_json(source_path)
            document = {
                "image": Path(source_doc.get("image", f"{stem}.png")).name,
                "elements": source_doc.get("elements", []),
            }
        else:
            document = default_document(stem)
        source = "prelabel"

    document["image"] = Path(document.get("image", f"{stem}.png")).name
    normalized_elements = [
        normalize_element(element, index + 1) for index, element in enumerate(document.get("elements", []))
    ]
    return {"image": document["image"], "elements": normalized_elements}, source


def save_document(stem: str, document: dict) -> dict:
    image_path = unlabeled_image_path(stem)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for stem: {stem}")
    if labeled_json_path(stem).exists():
        raise PermissionError(f"Document already saved for stem: {stem}")

    with Image.open(image_path) as image:
        image_size = image.size

    elements = [normalize_element(element, index + 1) for index, element in enumerate(document.get("elements", []))]
    saved_doc = {
        "image": f"{stem}.png",
        "elements": elements,
    }

    shutil.copy2(image_path, labeled_image_path(stem))
    labeled_json_path(stem).write_text(json.dumps(saved_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    update_hist_on_save(stem, elements, image_size)
    return saved_doc


def delete_stem(stem: str) -> None:
    if labeled_json_path(stem).exists():
        raise PermissionError(f"Document already saved for stem: {stem}")
    targets = [
        unlabeled_image_path(stem),
        unlabeled_json_path(stem),
        unlabeled_diff_path(stem),
        unlabeled_total_path(stem),
        labeled_image_path(stem),
        labeled_json_path(stem),
        labeled_total_path(stem),
    ]
    for path in targets:
        path.unlink(missing_ok=True)


class LabelRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(LABEL_DIR), **kwargs)

    def end_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, file_path: Path) -> None:
        content_type, _ = mimetypes.guess_type(str(file_path))
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/hist_top":
            limit_raw = query.get("limit", ["1000"])[0]
            try:
                limit = max(1, min(2000, int(limit_raw)))
            except ValueError:
                limit = 1000
            items = hist_top_items(limit)
            self.end_json({"items": items, "count": len(items)})
            return

        if path == "/api/entries":
            entries = [
                {
                    "stem": stem,
                    "image_name": f"{stem}.png",
                    "is_labeled": labeled_json_path(stem).exists(),
                }
                for stem in image_stems()
            ]
            self.end_json({"entries": entries})
            return

        if path == "/api/item":
            stem = query.get("stem", [""])[0]
            if not stem:
                self.end_json({"error": "Missing stem"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                document, source = load_document(stem)
            except Exception as exc:  # pragma: no cover
                self.end_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            load_counts = load_load_counts()
            load_counts[stem] = load_counts.get(stem, 0) + 1
            save_load_counts(load_counts)
            self.end_json(
                {
                    "stem": stem,
                    "source": source,
                    "document": document,
                    "load_count": load_counts[stem],
                    "image_url": f"/images/{urllib.parse.quote(f'{stem}.png')}",
                }
            )
            return

        if path.startswith("/images/"):
            image_name = Path(urllib.parse.unquote(path.removeprefix("/images/"))).name
            final_path = FINAL_LABEL_IMAGE_DIR / image_name
            source_path = PRELABEL_IMAGE_DIR / image_name
            file_path = final_path if final_path.exists() else source_path
            if not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                return
            self.serve_file(file_path)
            return

        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/delete":
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
                stem = str(payload.get("stem", "")).strip()
                if not stem:
                    raise ValueError("Missing stem")
                delete_stem(stem)
            except Exception as exc:
                self.end_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return

            self.end_json({"ok": True, "stem": stem})
            return

        if parsed.path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            stem = str(payload.get("stem", "")).strip()
            document = payload.get("document", {})
            if not stem:
                raise ValueError("Missing stem")
            saved_doc = save_document(stem, document)
        except Exception as exc:
            self.end_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        self.end_json(
            {
                "ok": True,
                "stem": stem,
                "image_path": str(labeled_image_path(stem)),
                "json_path": str(labeled_json_path(stem)),
                "document": saved_doc,
            }
        )


def main() -> None:
    ensure_dirs()
    server = ThreadingHTTPServer((HOST, PORT), LabelRequestHandler)
    print(f"Label server running at http://{HOST}:{PORT}")
    print(f"Source images: {PRELABEL_IMAGE_DIR}")
    print(f"Source data: {PRELABEL_DATA_DIR}")
    print(f"Save images: {FINAL_LABEL_IMAGE_DIR}")
    print(f"Save data: {FINAL_LABEL_DATA_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping label server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
