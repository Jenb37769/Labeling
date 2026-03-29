from __future__ import annotations

import json
import multiprocessing as mp
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

import cv2
import numpy as np

import process_video as pv


# Sampling config
FRAME_INTERVAL = 30

# Output config
PROCESS_TOTAL = pv.PROCESS_TOTAL
pv.PROCESS_TOTAL = PROCESS_TOTAL

# Parallel config
GPU_WORKER_COUNT = 2
TASK_QUEUE_SIZE = 8

# Fast prefilter config
ENABLE_FAST_PREFILTER = True
PREFILTER_WIDTH = 640
PREFILTER_CHROME_SIMILARITY_THRESHOLD = 0.995
PREFILTER_CANVAS_EDGE_CHANGE_THRESHOLD = 0.003


def load_state(json_path: Path) -> dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    target_height = max(1, int(height * target_width / width))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def build_prefilter_state(frame: np.ndarray) -> dict[str, np.ndarray]:
    small = resize_frame(frame, PREFILTER_WIDTH)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    top = gray[: int(height * pv.TOP_BAR_HEIGHT_RATIO), :]
    left = gray[:, : int(width * pv.LEFT_TOOLBAR_WIDTH_RATIO)]
    right = gray[:, int(width * pv.RIGHT_PANEL_START_RATIO) :]
    chrome = np.concatenate((top.flatten(), left.flatten(), right.flatten())).astype(np.float32)

    x1 = int(width * 0.10)
    x2 = int(width * 0.90)
    y1 = int(height * 0.12)
    y2 = int(height * 0.90)
    canvas = gray[y1:y2, x1:x2]
    canvas_edges = cv2.Canny(canvas, 80, 160)

    return {
        "chrome": chrome,
        "canvas_edges": canvas_edges,
    }


def fast_similarity(previous_state: dict[str, np.ndarray], current_state: dict[str, np.ndarray]) -> tuple[float, float]:
    chrome_diff = np.abs(previous_state["chrome"] - current_state["chrome"])
    chrome_similarity = max(0.0, 1.0 - float(chrome_diff.mean()) / 255.0)

    prev_edges = previous_state["canvas_edges"]
    curr_edges = current_state["canvas_edges"]
    if prev_edges.shape != curr_edges.shape:
        curr_edges = cv2.resize(curr_edges, (prev_edges.shape[1], prev_edges.shape[0]), interpolation=cv2.INTER_NEAREST)

    edge_change_ratio = float(np.count_nonzero(prev_edges != curr_edges)) / float(prev_edges.size)
    return chrome_similarity, edge_change_ratio


def should_skip_fast(previous_state: dict[str, np.ndarray], current_state: dict[str, np.ndarray]) -> tuple[bool, float, float]:
    chrome_similarity, edge_change_ratio = fast_similarity(previous_state, current_state)
    should_skip = (
        chrome_similarity >= PREFILTER_CHROME_SIMILARITY_THRESHOLD
        and edge_change_ratio <= PREFILTER_CANVAS_EDGE_CHANGE_THRESHOLD
    )
    return should_skip, chrome_similarity, edge_change_ratio


def parser_worker(task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    try:
        parser = pv.get_omniparser()
    except Exception as exc:  # pragma: no cover
        result_queue.put(
            {
                "kind": "worker_error",
                "error": f"worker_init_failed: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        task = task_queue.get()
        if task is None:
            break

        frame_index = task["frame_index"]
        image_name = task["image_name"]
        image_path = Path(task["image_path"])
        try:
            current_state, json_path, overlay_path = pv.parse_frame(image_path, parser, image_name=image_name)
            result_queue.put(
                {
                    "kind": "frame_result",
                    "frame_index": frame_index,
                    "image_name": image_name,
                    "image_path": str(image_path),
                    "json_path": str(json_path),
                    "overlay_path": str(overlay_path) if overlay_path is not None else "",
                    "element_count": current_state["element_count"],
                    "raw_element_count": current_state["raw_element_count"],
                }
            )
        except Exception as exc:  # pragma: no cover
            result_queue.put(
                {
                    "kind": "frame_error",
                    "frame_index": frame_index,
                    "image_name": image_name,
                    "image_path": str(image_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


def start_workers(task_queue: mp.Queue, result_queue: mp.Queue) -> list[mp.Process]:
    workers: list[mp.Process] = []
    for _ in range(GPU_WORKER_COUNT):
        worker = mp.Process(target=parser_worker, args=(task_queue, result_queue), daemon=True)
        worker.start()
        workers.append(worker)
    return workers


def stop_workers(workers: list[mp.Process], task_queue: mp.Queue) -> None:
    for _ in workers:
        task_queue.put(None)
    for worker in workers:
        worker.join()


def collect_results(result_queue: mp.Queue, expected_count: int) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    received = 0
    while received < expected_count:
        item = result_queue.get()
        if item["kind"] == "worker_error":
            raise RuntimeError(f"{item['error']}\n{item['traceback']}")
        if item["kind"] == "frame_error":
            raise RuntimeError(
                f"Frame parse failed for {item['image_name']}: {item['error']}\n{item['traceback']}"
            )
        results[item["frame_index"]] = item
        received += 1
    return results


def handle_result_item(
    item: dict[str, Any],
    previous_kept_state: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool, dict[str, Any] | None]:
    image_name = item["image_name"]
    image_path = Path(item["image_path"])
    json_path = Path(item["json_path"])
    overlay_path = Path(item["overlay_path"]) if item["overlay_path"] else None
    current_state = load_state(json_path)

    if previous_kept_state is None:
        final_paths = pv.finalize_keep(image_name, image_path, json_path, overlay_path=overlay_path)
        print(
            f"[KEEP] {image_name} first parsed frame "
            f"elements={item['element_count']} raw={item['raw_element_count']}"
        )
        print(f"         image={final_paths['image_path']}")
        print(f"         json={final_paths['json_path']}")
        if final_paths["overlay_path"] is not None:
            print(f"         total={final_paths['overlay_path']}")
        return current_state, True, None

    diff_data = pv.summarize_diff(previous_kept_state, current_state)
    diff_path = pv.write_diff(image_name, diff_data)
    summary = diff_data["summary"]

    if summary["keep"]:
        final_paths = pv.finalize_keep(image_name, image_path, json_path, diff_path=diff_path, overlay_path=overlay_path)
        print(
            f"[KEEP] {image_name} score={summary['total_change_score']:.3f} "
            f"high_conf={summary['high_conf_change_count']} reasons={','.join(summary['keep_reasons'])}"
        )
        print(f"         image={final_paths['image_path']}")
        print(f"         json={final_paths['json_path']}")
        print(f"         diff={final_paths['diff_path']}")
        if final_paths["overlay_path"] is not None:
            print(f"         total={final_paths['overlay_path']}")
        return current_state, True, diff_data

    pv.cleanup_outputs(image_path=image_path, json_path=json_path, diff_path=diff_path, overlay_path=overlay_path)
    print(
        f"[DROP] {image_name} score={summary['total_change_score']:.3f} "
        f"added={summary['added_count']} removed={summary['removed_count']} "
        f"modified={summary['modified_count']}"
    )
    return previous_kept_state, False, diff_data


def validate_result_item(item: dict[str, Any]) -> None:
    if item["kind"] == "worker_error":
        raise RuntimeError(f"{item['error']}\n{item['traceback']}")
    if item["kind"] == "frame_error":
        raise RuntimeError(
            f"Frame parse failed for {item['image_name']}: {item['error']}\n{item['traceback']}"
        )


def drain_available_results(result_queue: mp.Queue, pending_results: dict[int, dict[str, Any]]) -> None:
    while True:
        try:
            item = result_queue.get_nowait()
        except Empty:
            break
        validate_result_item(item)
        pending_results[item["frame_index"]] = item


def process_ready_results(
    submitted_indices: list[int],
    pending_results: dict[int, dict[str, Any]],
    next_result_pos: int,
    previous_kept_state: dict[str, Any] | None,
    kept_count: int,
    dropped_count: int,
) -> tuple[int, dict[str, Any] | None, int, int]:
    while next_result_pos < len(submitted_indices):
        frame_index = submitted_indices[next_result_pos]
        item = pending_results.pop(frame_index, None)
        if item is None:
            break
        previous_kept_state, kept, _ = handle_result_item(item, previous_kept_state)
        if kept:
            kept_count += 1
        else:
            dropped_count += 1
        next_result_pos += 1
    return next_result_pos, previous_kept_state, kept_count, dropped_count


def process_video(video_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    task_queue: mp.Queue = mp.Queue(maxsize=TASK_QUEUE_SIZE)
    result_queue: mp.Queue = mp.Queue()
    workers = start_workers(task_queue, result_queue)

    frame_index = 0
    sampled_count = 0
    submitted_count = 0
    prefiltered_count = 0
    submitted_indices: list[int] = []
    pending_results: dict[int, dict[str, Any]] = {}
    next_result_pos = 0
    previous_prefilter_state: dict[str, np.ndarray] | None = None
    previous_kept_state: dict[str, Any] | None = None
    kept_count = 0
    dropped_count = 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % FRAME_INTERVAL != 0:
                frame_index += 1
                continue

            sampled_count += 1
            current_prefilter_state = build_prefilter_state(frame)
            if ENABLE_FAST_PREFILTER and previous_prefilter_state is not None:
                skip_fast, chrome_similarity, edge_change_ratio = should_skip_fast(
                    previous_prefilter_state,
                    current_prefilter_state,
                )
                if skip_fast:
                    prefiltered_count += 1
                    print(
                        f"[PREFILTER-DROP] frame={frame_index:06d} "
                        f"chrome={chrome_similarity:.4f} canvas_edge_change={edge_change_ratio:.4f}"
                    )
                    frame_index += 1
                    continue

            previous_prefilter_state = current_prefilter_state
            timestamp_seconds = frame_index / fps if fps > 0 else 0.0
            image_name = f"{video_path.stem}_{frame_index:06d}_{timestamp_seconds:08.2f}s.png"
            image_path = pv.get_frame_image_path(image_name)
            cv2.imwrite(str(image_path), frame)

            task_queue.put(
                {
                    "frame_index": frame_index,
                    "image_name": image_name,
                    "image_path": str(image_path),
                }
            )
            submitted_indices.append(frame_index)
            submitted_count += 1
            drain_available_results(result_queue, pending_results)
            next_result_pos, previous_kept_state, kept_count, dropped_count = process_ready_results(
                submitted_indices,
                pending_results,
                next_result_pos,
                previous_kept_state,
                kept_count,
                dropped_count,
            )
            frame_index += 1
    finally:
        capture.release()
        stop_workers(workers, task_queue)

    while next_result_pos < len(submitted_indices):
        item = result_queue.get()
        validate_result_item(item)
        pending_results[item["frame_index"]] = item
        next_result_pos, previous_kept_state, kept_count, dropped_count = process_ready_results(
            submitted_indices,
            pending_results,
            next_result_pos,
            previous_kept_state,
            kept_count,
            dropped_count,
        )

    print()
    print(f"Processed video: {video_path.name}")
    print(f"Sampled frames: {sampled_count}")
    print(f"Submitted to OmniParser: {submitted_count}")
    print(f"Fast-prefilter dropped: {prefiltered_count}")
    print(f"Saved images: {kept_count}")
    print(f"Dropped after diff: {dropped_count}")
    print(f"Image output: {pv.FINAL_IMAGE_DIR}")
    print(f"Data output: {pv.FINAL_DATA_DIR}")
    if PROCESS_TOTAL:
        print(f"Total output: {pv.FINAL_TOTAL_DIR}")


def main() -> None:
    pv.ensure_dirs()
    source_video = pv.find_first_video()
    if source_video is None:
        print(f"No video found in {pv.VIDEO_UNLABEL_DIR}")
        return

    print(f"Processing video in place: {source_video}")
    process_video(source_video)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
