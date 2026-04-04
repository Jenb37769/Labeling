from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
PROCESSING_ROOT_DIR = ROOT_DIR / "processing"
FINAL_LABEL_DIR = PROCESSING_ROOT_DIR / "3.final_label"
FINAL_UNLABEL_DIR = PROCESSING_ROOT_DIR / "1.final_unlabel"

FINAL_LABEL_IMAGE_DIR = FINAL_LABEL_DIR / "ima"
FINAL_LABEL_DATA_DIR = FINAL_LABEL_DIR / "data"
FINAL_LABEL_TOTAL_DIR = FINAL_LABEL_DIR / "total"

FINAL_UNLABEL_IMAGE_DIR = FINAL_UNLABEL_DIR / "ima"
FINAL_UNLABEL_DATA_DIR = FINAL_UNLABEL_DIR / "data"
FINAL_UNLABEL_TOTAL_DIR = FINAL_UNLABEL_DIR / "total"


def ensure_dirs() -> None:
    FINAL_UNLABEL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_UNLABEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_UNLABEL_TOTAL_DIR.mkdir(parents=True, exist_ok=True)


def move_or_replace(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        return False
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    shutil.move(str(src), str(dst))
    return True


def remove_file(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        return True
    path.unlink()
    return True


def unlock_all(dry_run: bool) -> dict[str, int]:
    ensure_dirs()
    counts = {
        "total": 0,
        "moved_images": 0,
        "removed_labels": 0,
        "moved_totals": 0,
        "missing_images": 0,
        "missing_labels": 0,
    }

    for label_path in sorted(FINAL_LABEL_DATA_DIR.glob("*.json")):
        stem = label_path.stem
        counts["total"] += 1

        label_image = FINAL_LABEL_IMAGE_DIR / f"{stem}.png"
        target_image = FINAL_UNLABEL_IMAGE_DIR / f"{stem}.png"
        if move_or_replace(label_image, target_image, dry_run):
            counts["moved_images"] += 1
        else:
            counts["missing_images"] += 1

        label_total = FINAL_LABEL_TOTAL_DIR / f"{stem}_overlay.png"
        target_total = FINAL_UNLABEL_TOTAL_DIR / f"{stem}_overlay.png"
        if move_or_replace(label_total, target_total, dry_run):
            counts["moved_totals"] += 1

        if remove_file(label_path, dry_run):
            counts["removed_labels"] += 1
        else:
            counts["missing_labels"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Unlock saved labels back to unlabel state.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without modifying files.")
    args = parser.parse_args()

    counts = unlock_all(dry_run=args.dry_run)
    prefix = "[DRY RUN]" if args.dry_run else "[DONE]"
    print(
        f"{prefix} total={counts['total']} moved_images={counts['moved_images']} "
        f"removed_labels={counts['removed_labels']} moved_totals={counts['moved_totals']} "
        f"missing_images={counts['missing_images']} missing_labels={counts['missing_labels']}"
    )


if __name__ == "__main__":
    main()
