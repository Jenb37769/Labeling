from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


ROOT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT_DIR / "script"
DEFAULT_INPUT_DIR = ROOT_DIR / "output"
DEFAULT_RESULT_DIR = ROOT_DIR / "question_output"
REQUEST_RETRY_COUNT = 3
REQUEST_RETRY_SLEEP_SECONDS = 2.0

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prelabel as single_prelabel  # noqa: E402
import test_prelabel_batch as batch_prelabel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask the current LLM with the exported inputs from output/prelabel and output/test_prelabel_batch, "
            "then report prompt/completion/total token usage for each version."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--model", type=str, default=os.environ.get("PRELABEL_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", type=str, default=os.environ.get("PRELABEL_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("PRELABEL_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--http-referer", type=str, default=os.environ.get("PRELABEL_HTTP_REFERER", ""))
    parser.add_argument("--app-title", type=str, default=os.environ.get("PRELABEL_APP_TITLE", "grounding-question"))
    args = parser.parse_args()
    args.model = str(args.model or "").strip()
    args.base_url = str(args.base_url or "").strip()
    args.api_key = str(args.api_key or "").strip()
    args.http_referer = str(args.http_referer or "").strip()
    args.app_title = str(args.app_title or "").strip()
    return args


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def image_path_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".png":
        mime_type = "image/png"
    else:
        raise ValueError(f"Unsupported image type: {path}")
    payload = path.read_bytes()
    import base64

    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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
    system_prompt: str,
    prompt_text: str,
    image_paths: list[Path],
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    http_referer: str,
    app_title: str,
) -> tuple[dict[str, Any], str]:
    if not api_key:
        raise ValueError("Missing PRELABEL_API_KEY or --api-key.")

    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for image_path in image_paths:
        user_content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path)}})

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "grounding-question/1.0",
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
            return response_json, raw_text
        except Exception as exc:
            last_error = exc
            if attempt >= REQUEST_RETRY_COUNT:
                raise RuntimeError(f"Question request failed after {REQUEST_RETRY_COUNT} attempts: {exc}") from exc
            print(
                f"[RETRY] attempt={attempt}/{REQUEST_RETRY_COUNT} failed; retrying in "
                f"{REQUEST_RETRY_SLEEP_SECONDS:.1f}s"
            )
            time.sleep(REQUEST_RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"Question request failed: {last_error}")


def load_prelabel_input(input_dir: Path) -> tuple[str, list[Path], dict[str, Any]]:
    folder = input_dir / "prelabel"
    prompt_text = load_text(folder / "prompt.txt")
    meta = load_json(folder / "meta.json")
    image_paths = [Path(meta["context_image"]), Path(meta["crop_image"])]
    return prompt_text, image_paths, meta


def load_batch_input(input_dir: Path) -> tuple[str, list[Path], dict[str, Any]]:
    folder = input_dir / "test_prelabel_batch"
    prompt_text = load_text(folder / "prompt.txt")
    meta = load_json(folder / "meta.json")
    image_paths = [Path(meta["overview_image"])] + [Path(path) for path in meta.get("crop_images", [])]
    return prompt_text, image_paths, meta


def usage_summary(response_json: dict[str, Any]) -> dict[str, Any]:
    usage = response_json.get("usage", {})
    completion_details = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}
    prompt_details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_text_tokens": prompt_details.get("text_tokens"),
        "prompt_image_tokens": prompt_details.get("image_tokens"),
        "completion_reasoning_tokens": completion_details.get("reasoning_tokens"),
        "finish_reason": response_json.get("choices", [{}])[0].get("finish_reason"),
    }


def run_one(
    *,
    name: str,
    system_prompt: str,
    prompt_text: str,
    image_paths: list[Path],
    meta: dict[str, Any],
    args: argparse.Namespace,
    result_dir: Path,
) -> dict[str, Any]:
    version_dir = result_dir / name
    version_dir.mkdir(parents=True, exist_ok=True)

    response_json, raw_text = call_openai_compatible_api(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        system_prompt=system_prompt,
        prompt_text=prompt_text,
        image_paths=image_paths,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        http_referer=args.http_referer,
        app_title=args.app_title,
    )

    summary = usage_summary(response_json)
    write_json(version_dir / "meta.json", meta)
    write_text(version_dir / "prompt.txt", prompt_text)
    write_json(version_dir / "response_raw.json", response_json)
    write_text(version_dir / "response_text.txt", raw_text)
    write_json(version_dir / "usage.json", summary)

    print(
        f"[QUESTION] {name} "
        f"images={len(image_paths)} "
        f"prompt_tokens={summary.get('prompt_tokens', '-')} "
        f"completion_tokens={summary.get('completion_tokens', '-')} "
        f"total_tokens={summary.get('total_tokens', '-')} "
        f"reasoning_tokens={summary.get('completion_reasoning_tokens', '-')}"
    )
    return summary


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    prelabel_prompt, prelabel_images, prelabel_meta = load_prelabel_input(args.input_dir)
    batch_prompt, batch_images, batch_meta = load_batch_input(args.input_dir)

    prelabel_summary = run_one(
        name="prelabel",
        system_prompt=single_prelabel.SYSTEM_PROMPT,
        prompt_text=prelabel_prompt,
        image_paths=prelabel_images,
        meta=prelabel_meta,
        args=args,
        result_dir=result_dir,
    )
    batch_summary = run_one(
        name="test_prelabel_batch",
        system_prompt=batch_prelabel.SYSTEM_PROMPT,
        prompt_text=batch_prompt,
        image_paths=batch_images,
        meta=batch_meta,
        args=args,
        result_dir=result_dir,
    )

    write_json(
        result_dir / "summary.json",
        {
            "model": args.model,
            "base_url": args.base_url,
            "prelabel": prelabel_summary,
            "test_prelabel_batch": batch_summary,
        },
    )

    print(f"[DONE] result_dir={result_dir}")


if __name__ == "__main__":
    main()
