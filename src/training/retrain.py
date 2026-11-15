"""Run an isolated retraining job and promote it only when it beats the active model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_baseline import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_MODEL_DIR,
    DEFAULT_REPORT_DIR,
    build_dataloaders,
    build_model,
    evaluate,
    get_device,
)

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "splitted_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a dataset, train an isolated candidate, compare and promote it."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs" / "retraining")
    parser.add_argument(
        "--model-name", choices=("resnet18", "mobilenet_v3_small"), default="resnet18"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--min-f1-improvement",
        type=float,
        default=0.0,
        help="Required validation F1 improvement over the active model (default: strictly better).",
    )
    parser.add_argument(
        "--max-recall-drop",
        type=float,
        default=0.0,
        help="Maximum allowed validation recall drop for non_edible (default: no drop).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("class_names") != CLASS_NAMES or "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint is incompatible with this classifier: {path}")
    return checkpoint


@torch.no_grad()
def score_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    image_size: int,
    batch_size: int,
) -> dict[str, float | None]:
    device = get_device()
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(
        model_name=checkpoint["model_name"],
        pretrained=False,
        freeze_backbone=False,
        num_classes=len(CLASS_NAMES),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loaders, _ = build_dataloaders(
        data_dir=data_dir,
        image_size=int(checkpoint.get("image_size", image_size)),
        batch_size=batch_size,
        num_workers=0,
    )
    result = evaluate(model, loaders["val"], nn.CrossEntropyLoss(), device)
    return {
        "accuracy": float(result["accuracy"]),
        "precision_non_edible": float(result["precision"]),
        "recall_non_edible": float(result["recall"]),
        "f1_non_edible": float(result["f1"]),
        "roc_auc": None if result["roc_auc"] is None else float(result["roc_auc"]),
    }


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def promote(
    candidate: Path,
    candidate_metrics: Path,
    active_model: Path,
    active_metrics: Path,
) -> None:
    """Replace the active pair, restoring the previous pair if either write fails."""
    backup_dir = candidate.parent / "promotion_backup"
    old_model = backup_dir / "active_model.pt"
    old_metrics = backup_dir / "active_metrics.json"
    had_model, had_metrics = active_model.exists(), active_metrics.exists()
    if had_model:
        shutil.copy2(active_model, old_model)
    if had_metrics:
        shutil.copy2(active_metrics, old_metrics)
    try:
        atomic_copy(candidate_metrics, active_metrics)
        atomic_copy(candidate, active_model)
    except OSError:
        if had_model:
            atomic_copy(old_model, active_model)
        else:
            active_model.unlink(missing_ok=True)
        if had_metrics:
            atomic_copy(old_metrics, active_metrics)
        else:
            active_metrics.unlink(missing_ok=True)
        raise


def run() -> int:
    args = parse_args()
    if not 0 <= args.min_f1_improvement <= 1:
        raise ValueError("--min-f1-improvement must be between 0 and 1")
    if not 0 <= args.max_recall_drop <= 1:
        raise ValueError("--max-recall-drop must be between 0 and 1")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = (args.run_root / run_id).resolve()
    raw_dir = args.raw_dir.resolve()
    data_dir = run_dir / "data" / "processed"
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    run_dir.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "raw_dir": str(raw_dir),
        "parameters": {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "image_size": args.image_size,
            "seed": args.seed,
            "pretrained": args.pretrained,
            "min_f1_improvement": args.min_f1_improvement,
            "max_recall_drop": args.max_recall_drop,
        },
        "status": "running",
    }
    manifest_path = run_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        prepare = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "data" / "prepare_dataset.py"),
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(data_dir),
            "--overwrite",
        ]
        subprocess.run(prepare, cwd=PROJECT_ROOT, check=True)
        data_manifest = data_dir / "manifest.csv"
        manifest["dataset_manifest_sha256"] = sha256_file(data_manifest)

        train = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "training" / "train_baseline.py"),
            "--data-dir",
            str(data_dir),
            "--report-dir",
            str(report_dir),
            "--model-dir",
            str(model_dir),
            "--model-name",
            args.model_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--image-size",
            str(args.image_size),
            "--seed",
            str(args.seed),
        ]
        if args.pretrained:
            train.append("--pretrained")
        subprocess.run(train, cwd=PROJECT_ROOT, check=True)

        candidate_path = model_dir / f"baseline_{args.model_name}.pt"
        candidate_metrics_path = report_dir / "metrics.json"
        candidate_scores = score_checkpoint(
            candidate_path, data_dir, args.image_size, args.batch_size
        )
        active_model = DEFAULT_MODEL_DIR / "baseline_resnet18.pt"
        active_metrics = DEFAULT_REPORT_DIR / "best_baseline.json"
        incumbent_scores = None
        if active_model.exists():
            incumbent_scores = score_checkpoint(
                active_model, data_dir, args.image_size, args.batch_size
            )

        incumbent_f1 = incumbent_scores["f1_non_edible"] if incumbent_scores else -1.0
        f1_threshold = incumbent_f1 + args.min_f1_improvement
        recall_threshold = (
            incumbent_scores["recall_non_edible"] if incumbent_scores else 0.0
        ) - args.max_recall_drop
        accepted = (
            candidate_scores["f1_non_edible"] > f1_threshold
            and candidate_scores["recall_non_edible"] >= recall_threshold
        )
        decision = {
            "evaluation_split": "val",
            "incumbent": incumbent_scores,
            "candidate": candidate_scores,
            "accepted": accepted,
            "criteria": {
                "candidate_f1_must_exceed": f1_threshold,
                "candidate_recall_must_be_at_least": recall_threshold,
            },
        }
        manifest["decision"] = decision
        manifest["status"] = "promoted" if accepted else "rejected"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        (report_dir / "promotion_evaluation.json").write_text(
            json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        metrics_payload = json.loads(candidate_metrics_path.read_text(encoding="utf-8"))
        metrics_payload["retraining"] = decision
        candidate_metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if accepted:
            promote(candidate_path, candidate_metrics_path, active_model, active_metrics)
            print(f"Candidate promoted. Restart the API to load: {active_model}")
        else:
            print("Candidate rejected; active model was left unchanged.")
        print(f"Run artifacts: {run_dir}")
        return 0 if accepted else 2
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        raise
    finally:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    raise SystemExit(run())
