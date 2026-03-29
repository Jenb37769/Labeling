from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
TARGET_DIRS = [
    ROOT_DIR / "final_unlabel",
    ROOT_DIR / "image",
    ROOT_DIR / "process",
]
DELETE_SUFFIXES = {
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
}


def clear_directory_contents(directory: Path) -> tuple[int, int]:
    removed_files = 0

    if not directory.exists():
        return removed_files, 0

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in DELETE_SUFFIXES:
            path.unlink(missing_ok=True)
            removed_files += 1

    return removed_files, 0


def main() -> None:
    total_files = 0
    total_dirs = 0

    for directory in TARGET_DIRS:
        removed_files, removed_dirs = clear_directory_contents(directory)
        total_files += removed_files
        total_dirs += removed_dirs
        print(f"Cleared {directory}: files={removed_files}, empty_subdirs_removed={removed_dirs}")

    print(f"Done. Total files removed: {total_files}, empty subdirectories removed: {total_dirs}")


if __name__ == "__main__":
    main()
