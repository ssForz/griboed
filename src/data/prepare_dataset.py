from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "splitted_dataset"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ("train", "val", "test")
SOURCE_LABELS = ("edible", "poisonous")
TARGET_LABELS = {
    "edible": "edible",
    "poisonous": "non_edible",
}
CLASS_TO_IDX = {
    "edible": 0,
    "non_edible": 1,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    split: str
    source_label: str
    label: str
    label_id: int
    species: str
    source_path: str
    processed_path: str
    width: int
    height: int
    mode: str
    image_format: str
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare mushroom images for binary edibility classification."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--resize-mode",
        choices=("pad", "stretch"),
        default="pad",
        help="pad keeps aspect ratio; stretch resizes directly to a square.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output directory before writing new files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate images and print metadata without writing processed files.",
    )
    return parser.parse_args()


def iter_image_paths(raw_dir: Path) -> list[tuple[str, str, Path]]:
    image_paths: list[tuple[str, str, Path]] = []

    for split in SPLITS:
        for source_label in SOURCE_LABELS:
            class_dir = raw_dir / split / source_label
            if not class_dir.exists():
                raise FileNotFoundError(f"Expected directory not found: {class_dir}")

            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append((split, source_label, path))

    return image_paths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_species(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")

    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])

    return stem


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_image_metadata(path: Path) -> tuple[int, int, str, str]:
    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format or path.suffix.lower().lstrip(".").upper()

        return width, height, mode, image_format
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Image cannot be opened: {path}") from exc


def preprocess_image(
    source_path: Path,
    output_path: Path,
    image_size: int,
    resize_mode: str,
    jpeg_quality: int,
) -> None:
    with Image.open(source_path) as image:
        image = image.convert("RGB")

        if resize_mode == "pad":
            image = ImageOps.pad(
                image,
                (image_size, image_size),
                method=Image.Resampling.BICUBIC,
                color=(0, 0, 0),
                centering=(0.5, 0.5),
            )
        else:
            image = image.resize(
                (image_size, image_size),
                resample=Image.Resampling.BICUBIC,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)


def prepare_output_dir(output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)

    for split in SPLITS:
        for label in CLASS_TO_IDX:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)


def build_record(
    split: str,
    source_label: str,
    source_path: Path,
    output_dir: Path,
    image_index: int,
) -> ImageRecord:
    label = TARGET_LABELS[source_label]
    relative_output_path = (
        Path(split)
        / label
        / f"{label}_{image_index:06d}{source_path.suffix.lower()}"
    ).with_suffix(".jpg")
    width, height, mode, image_format = read_image_metadata(source_path)

    return ImageRecord(
        split=split,
        source_label=source_label,
        label=label,
        label_id=CLASS_TO_IDX[label],
        species=extract_species(source_path),
        source_path=project_relative(source_path),
        processed_path=project_relative(output_dir / relative_output_path),
        width=width,
        height=height,
        mode=mode,
        image_format=image_format,
        sha256=file_sha256(source_path),
    )


def write_manifest(path: Path, records: list[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_metadata(
    output_dir: Path,
    raw_dir: Path,
    records: list[ImageRecord],
    image_size: int,
    resize_mode: str,
    jpeg_quality: int,
) -> None:
    split_counts = {
        split: {
            label: sum(
                record.split == split and record.label == label
                for record in records
            )
            for label in CLASS_TO_IDX
        }
        for split in SPLITS
    }
    class_counts = {
        label: sum(record.label == label for record in records)
        for label in CLASS_TO_IDX
    }
    duplicate_groups = {}

    for record in records:
        duplicate_groups.setdefault(record.sha256, []).append(record.source_path)

    duplicates = {
        digest: paths
        for digest, paths in duplicate_groups.items()
        if len(paths) > 1
    }
    source_path_to_split = {
        record.source_path: record.split
        for record in records
    }
    cross_split_duplicates = {
        digest: paths
        for digest, paths in duplicates.items()
        if len({source_path_to_split[path] for path in paths}) > 1
    }
    metadata = {
        "task": "binary image classification",
        "source_dir": project_relative(raw_dir),
        "output_dir": project_relative(output_dir),
        "splits": list(SPLITS),
        "source_labels": list(SOURCE_LABELS),
        "target_labels": TARGET_LABELS,
        "class_to_idx": CLASS_TO_IDX,
        "image_size": [image_size, image_size],
        "resize_mode": resize_mode,
        "jpeg_quality": jpeg_quality,
        "total_images": len(records),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "duplicate_hash_count": len(duplicates),
        "duplicate_image_count": sum(len(paths) for paths in duplicates.values()),
        "cross_split_duplicate_hash_count": len(cross_split_duplicates),
        "cross_split_duplicate_image_count": sum(
            len(paths) for paths in cross_split_duplicates.values()
        ),
    }

    with (output_dir / "dataset_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    if duplicates:
        with (output_dir / "duplicate_images.json").open("w", encoding="utf-8") as file:
            json.dump(duplicates, file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    image_paths = iter_image_paths(raw_dir)
    if not image_paths:
        raise RuntimeError(f"No image files found in {raw_dir}")

    prepare_output_dir(output_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    records: list[ImageRecord] = []
    per_target_counter = {split: {label: 0 for label in CLASS_TO_IDX} for split in SPLITS}
    broken_images: list[str] = []

    for split, source_label, source_path in tqdm(image_paths, desc="Preparing images"):
        label = TARGET_LABELS[source_label]
        image_index = per_target_counter[split][label]

        try:
            record = build_record(split, source_label, source_path, output_dir, image_index)
        except ValueError:
            broken_images.append(str(source_path.relative_to(PROJECT_ROOT)))
            continue

        if not args.dry_run:
            preprocess_image(
                source_path=source_path,
                output_path=PROJECT_ROOT / record.processed_path,
                image_size=args.image_size,
                resize_mode=args.resize_mode,
                jpeg_quality=args.jpeg_quality,
            )

        records.append(record)
        per_target_counter[split][label] += 1

    if not records:
        raise RuntimeError("No valid images found.")

    if not args.dry_run:
        all_manifest_path = output_dir / "manifest.csv"
        write_manifest(all_manifest_path, records)

        for split in SPLITS:
            split_records = [record for record in records if record.split == split]
            write_manifest(output_dir / f"{split}_manifest.csv", split_records)

        write_metadata(
            output_dir=output_dir,
            raw_dir=raw_dir,
            records=records,
            image_size=args.image_size,
            resize_mode=args.resize_mode,
            jpeg_quality=args.jpeg_quality,
        )

        if broken_images:
            with (output_dir / "broken_images.json").open("w", encoding="utf-8") as file:
                json.dump(broken_images, file, indent=2, ensure_ascii=False)

    print(f"Valid images: {len(records)}")
    print(f"Broken images: {len(broken_images)}")
    print("Class distribution:")
    for label in CLASS_TO_IDX:
        count = sum(record.label == label for record in records)
        print(f"  {label}: {count}")

    print("Split distribution:")
    for split in SPLITS:
        counts = {
            label: sum(record.split == split and record.label == label for record in records)
            for label in CLASS_TO_IDX
        }
        print(f"  {split}: {counts}")

    if args.dry_run:
        print("Dry run complete. No files were written.")
    else:
        print(f"Processed dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
