from __future__ import annotations

import json
import mimetypes
import shutil
import sys
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT_DIR / "label"
FINAL_UNLABEL_IMAGE_DIR = ROOT_DIR / "final_unlabel" / "ima"
FINAL_UNLABEL_DATA_DIR = ROOT_DIR / "final_unlabel" / "data"
FINAL_UNLABEL_TOTAL_DIR = ROOT_DIR / "final_unlabel" / "total"
FINAL_LABEL_IMAGE_DIR = ROOT_DIR / "final_label" / "ima"
FINAL_LABEL_DATA_DIR = ROOT_DIR / "final_label" / "data"
FINAL_LABEL_TOTAL_DIR = ROOT_DIR / "final_label" / "total"
HOST = "127.0.0.1"
PORT = 8765


def ensure_dirs() -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_LABEL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_LABEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_LABEL_TOTAL_DIR.mkdir(parents=True, exist_ok=True)


def image_stems() -> list[str]:
    stems = []
    for image_path in sorted(FINAL_UNLABEL_IMAGE_DIR.glob("*.png")):
        if image_path.name.startswith("."):
            continue
        stems.append(image_path.stem)
    return stems


def unlabeled_json_path(stem: str) -> Path:
    return FINAL_UNLABEL_DATA_DIR / f"{stem}_omniparser.json"


def labeled_json_path(stem: str) -> Path:
    return FINAL_LABEL_DATA_DIR / f"{stem}.json"


def labeled_image_path(stem: str) -> Path:
    return FINAL_LABEL_IMAGE_DIR / f"{stem}.png"


def unlabeled_image_path(stem: str) -> Path:
    return FINAL_UNLABEL_IMAGE_DIR / f"{stem}.png"


def unlabeled_diff_path(stem: str) -> Path:
    return FINAL_UNLABEL_DATA_DIR / f"{stem}_diff.json"


def unlabeled_total_path(stem: str) -> Path:
    return FINAL_UNLABEL_TOTAL_DIR / f"{stem}_overlay.png"


def labeled_total_path(stem: str) -> Path:
    return FINAL_LABEL_TOTAL_DIR / f"{stem}_overlay.png"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        source = "final_unlabel"

    document["image"] = Path(document.get("image", f"{stem}.png")).name
    normalized_elements = [
        normalize_element(element, index + 1) for index, element in enumerate(document.get("elements", []))
    ]
    return {"image": document["image"], "elements": normalized_elements}, source


def save_document(stem: str, document: dict) -> dict:
    image_path = unlabeled_image_path(stem)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for stem: {stem}")

    elements = [normalize_element(element, index + 1) for index, element in enumerate(document.get("elements", []))]
    saved_doc = {
        "image": f"{stem}.png",
        "elements": elements,
    }

    shutil.copy2(image_path, labeled_image_path(stem))
    labeled_json_path(stem).write_text(json.dumps(saved_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return saved_doc


def delete_stem(stem: str) -> None:
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
            self.end_json(
                {
                    "stem": stem,
                    "source": source,
                    "document": document,
                    "image_url": f"/images/{urllib.parse.quote(f'{stem}.png')}",
                }
            )
            return

        if path.startswith("/images/"):
            image_name = Path(urllib.parse.unquote(path.removeprefix("/images/"))).name
            final_path = FINAL_LABEL_IMAGE_DIR / image_name
            source_path = FINAL_UNLABEL_IMAGE_DIR / image_name
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
    print(f"Source images: {FINAL_UNLABEL_IMAGE_DIR}")
    print(f"Source data: {FINAL_UNLABEL_DATA_DIR}")
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
