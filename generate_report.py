from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from cache_config import configure_external_caches


configure_external_caches()

PROJECT_ROOT = Path("/home/ubuntu/ml-platform/other-projects/lica-score")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lica_score.data import jsonl_read  # noqa: E402
from lica_score.model import LicaScorer, pairwise_logistic_loss  # noqa: E402
from lica_score.train import candidate_key, condition_key, evaluate_records  # noqa: E402
from lica_score.train_lora import cache_eval_embeddings, load_qwen_with_lora  # noqa: E402


MODEL_CONFIGS = [
    {
        "id": "qwen2b",
        "name": "Qwen3-VL-Embedding-2B",
        "model_name": "Qwen/Qwen3-VL-Embedding-2B",
        "embedding_dim": 2048,
        "output_dir": PROJECT_ROOT / "outputs/model_compare/qwen2b_lora_cosine_5epoch",
    },
    {
        "id": "qwen8b",
        "name": "Qwen3-VL-Embedding-8B",
        "model_name": "Qwen/Qwen3-VL-Embedding-8B",
        "embedding_dim": 4096,
        "output_dir": PROJECT_ROOT / "outputs/model_compare/qwen8b_lora_cosine_5epoch",
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_metrics(output_dir: Path) -> list[dict[str, Any]]:
    metrics_path = output_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    return load_jsonl(metrics_path)


def sanitize_asset_name(path: str) -> str:
    source = Path(path)
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    return f"{source.stem}-{digest}{source.suffix or '.png'}"


def copy_asset(path: str, assets_dir: Path, cache: dict[str, str]) -> str:
    if path in cache:
        return cache[path]
    assets_dir.mkdir(parents=True, exist_ok=True)
    name = sanitize_asset_name(path)
    dest = assets_dir / name
    if not dest.exists():
        shutil.copy2(path, dest)
    rel = f"assets/{name}"
    cache[path] = rel
    return rel


def validation_loss(
    scorer: LicaScorer,
    records: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> float:
    losses: list[float] = []
    scorer.eval()
    with torch.no_grad():
        for record in records:
            cond = torch.from_numpy(embeddings[condition_key(record)]).unsqueeze(0).to(device)
            pos = torch.from_numpy(embeddings[candidate_key(record["positive"])]).unsqueeze(0).to(device)
            neg = torch.from_numpy(embeddings[candidate_key(record["negative"])]).unsqueeze(0).to(device)
            loss = pairwise_logistic_loss(scorer(cond, pos), scorer(cond, neg))
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def score_candidates(
    scorer: LicaScorer,
    records: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"task": "text_to_svg", "candidates": {}})
    scorer.eval()
    for record in records:
        group = grouped[record["group_id"]]
        group["task"] = record["task"]
        group["candidates"][candidate_key(record["positive"])] = record["positive"]
        group["candidates"][candidate_key(record["negative"])] = record["negative"]

    scores: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for group_id, payload in grouped.items():
            cond = torch.from_numpy(embeddings[f"condition:{payload['task']}:{group_id}"]).unsqueeze(0).to(device)
            scores[group_id] = {}
            for key, candidate in payload["candidates"].items():
                cand = torch.from_numpy(embeddings[key]).unsqueeze(0).to(device)
                score = scorer(cond, cand)
                if scorer.scorer_type == "cosine":
                    scale = scorer.logit_scale.clamp(max=math.log(100.0)).exp()
                    score = score / scale
                scores[group_id][candidate["source"]] = float(score.item())
    return scores


def gt_ai_summary(records: list[dict[str, Any]], scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    correct = 0
    total = 0
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for record in records:
        group_scores = scores[record["group_id"]]
        pos_source = record["positive"]["source"]
        neg_source = record["negative"]["source"]
        is_correct = group_scores[pos_source] > group_scores[neg_source]
        correct += int(is_correct)
        total += 1
        by_source[neg_source]["correct"] += int(is_correct)
        by_source[neg_source]["total"] += 1
    by_source_payload = {
        src: {
            "correct": vals["correct"],
            "total": vals["total"],
            "accuracy": vals["correct"] / vals["total"] if vals["total"] else 0.0,
        }
        for src, vals in sorted(by_source.items())
    }
    return {
        "accuracy": correct / total if total else 0.0,
        "confusion": {
            "true_gt_pred_gt": correct,
            "true_gt_pred_ai": total - correct,
            "total": total,
        },
        "by_source": by_source_payload,
    }


def make_base_qwen(model_name: str, device: torch.device):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name,
        device=str(device),
        model_kwargs={"key_mapping": {"^model\\.": ""}},
    )


def make_lora_qwen(config: dict[str, Any], epoch: int, device: torch.device):
    lora_path = config["output_dir"] / f"qwen_lora_state_epoch_{epoch:03d}.pt"
    if not lora_path.exists():
        raise FileNotFoundError(lora_path)
    checkpoint = torch.load(lora_path, map_location="cpu", weights_only=False)
    lora_config = checkpoint["lora_config"]
    args = SimpleNamespace(
        model_name=config["model_name"],
        device=str(device),
        gradient_checkpointing=False,
        lora_r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        lora_target_modules=",".join(lora_config["target_modules"]),
    )
    model = load_qwen_with_lora(args)
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model[0].auto_model, checkpoint["lora_state"])
    model.eval()
    return model


def load_scorer(path: Path, device: torch.device) -> LicaScorer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["scorer_config"])
    scorer = LicaScorer(**config)
    scorer.load_state_dict(checkpoint["scorer_state"])
    scorer.to(device)
    scorer.eval()
    return scorer


def release(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_variant(
    *,
    variant_id: str,
    label: str,
    model_config: dict[str, Any],
    records: list[dict[str, Any]],
    device: torch.device,
    epoch: int,
    train_loss: float | None,
    is_base: bool,
) -> dict[str, Any]:
    if is_base:
        qwen = make_base_qwen(model_config["model_name"], device)
        scorer = LicaScorer(embedding_dim=model_config["embedding_dim"], scorer_type="cosine").to(device)
    else:
        qwen = make_lora_qwen(model_config, epoch, device)
        scorer = load_scorer(model_config["output_dir"] / f"scorer_epoch_{epoch:03d}.pt", device)

    embeddings = cache_eval_embeddings(
        qwen,
        records,
        device=device,
        embedding_dim=model_config["embedding_dim"],
        max_svg_chars=2048,
        include_svg_text=False,
    )
    metrics = evaluate_records(scorer, records, embeddings, device=device)
    scores = score_candidates(scorer, records, embeddings, device=device)
    payload = {
        "id": variant_id,
        "label": label,
        "model_id": model_config["id"],
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss(scorer, records, embeddings, device=device),
        "metrics": metrics,
        "summary": gt_ai_summary(records, scores),
        "score_display": "raw_cosine" if scorer.scorer_type == "cosine" else scorer.scorer_type,
        "scores": scores,
    }
    release(qwen, scorer)
    return payload


def build_group_payload(
    records: list[dict[str, Any]],
    variant_scores: dict[str, dict[str, dict[str, float]]],
    *,
    assets_dir: Path,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    asset_cache: dict[str, str] = {}
    for record in records:
        group = groups.setdefault(
            record["group_id"],
            {
                "group_id": record["group_id"],
                "bucket": record.get("bucket")
                or record.get("positive", {}).get("bucket")
                or record.get("negative", {}).get("bucket", ""),
                "prompt": record["condition"]["text_prompt"],
                "candidates": {},
            },
        )
        for side in ("positive", "negative"):
            candidate = record[side]
            group["candidates"][candidate["source"]] = {
                "source": candidate["source"],
                "image": copy_asset(candidate["render_path"], assets_dir, asset_cache),
            }

    ordered_sources = ["gt", "claude", "gemini", "gpt-5.2"]
    payload: list[dict[str, Any]] = []
    for group_id in sorted(groups):
        group = groups[group_id]
        entries = []
        for source in ordered_sources:
            if source not in group["candidates"]:
                continue
            entries.append(
                {
                    **group["candidates"][source],
                    "scores": {
                        variant_id: scores[group_id][source]
                        for variant_id, scores in variant_scores.items()
                        if group_id in scores and source in scores[group_id]
                    },
                }
            )
        winners = {}
        for variant_id, scores in variant_scores.items():
            if group_id not in scores:
                continue
            winners[variant_id] = max(scores[group_id].items(), key=lambda item: item[1])[0]
        payload.append(
            {
                "group_id": group_id,
                "bucket": group["bucket"],
                "prompt": group["prompt"],
                "winners": winners,
                "entries": entries,
            }
        )
    return payload


def build_ai_comparisons(groups: list[dict[str, Any]], variant_scores: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, Any]]:
    comparisons = []
    pairs = [("gpt-5.2", "claude"), ("claude", "gemini"), ("gpt-5.2", "gemini")]
    for group in groups:
        entries = {entry["source"]: entry for entry in group["entries"]}
        for left, right in pairs:
            if left not in entries or right not in entries:
                continue
            winners = {}
            for variant_id, scores in variant_scores.items():
                left_score = scores[group["group_id"]][left]
                right_score = scores[group["group_id"]][right]
                winners[variant_id] = left if left_score >= right_score else right
            comparisons.append(
                {
                    "group_id": group["group_id"],
                    "bucket": group["bucket"],
                    "pair": f"{left} vs {right}",
                    "left": entries[left],
                    "right": entries[right],
                    "winners": winners,
                }
            )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, default=PROJECT_ROOT / "data/processed/svg_data_v1")
    parser.add_argument("--web-root", type=Path, default=Path("/home/ubuntu/lica-score-web"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = jsonl_read(args.pairs_dir / "eval_pairs.jsonl")
    device = torch.device(args.device)
    assets_dir = args.web_root / "assets"

    models = []
    variants = []
    variant_scores: dict[str, dict[str, dict[str, float]]] = {}
    for config in MODEL_CONFIGS:
        logged = read_metrics(config["output_dir"])
        model_variants = []
        model_variant_payloads: dict[str, dict[str, Any]] = {}

        base_id = f"{config['id']}_base"
        base_variant = evaluate_variant(
            variant_id=base_id,
            label=f"{config['name']} Base",
            model_config=config,
            records=records,
            device=device,
            epoch=0,
            train_loss=None,
            is_base=True,
        )
        base_variant_public = {k: v for k, v in base_variant.items() if k != "scores"}
        variants.append(base_variant_public)
        model_variant_payloads[base_id] = base_variant_public
        variant_scores[base_id] = base_variant["scores"]
        model_variants.append(base_id)

        epoch_metrics = [
            {
                "epoch": 0,
                "train_loss": None,
                "validation_loss": base_variant["validation_loss"],
                "pairwise_accuracy": base_variant["metrics"]["pairwise_accuracy"],
                "hit_at_1": base_variant["metrics"]["hit_at_1"],
                "mrr": base_variant["metrics"]["mrr"],
                "mean_margin": base_variant["metrics"]["mean_margin"],
            }
        ]

        for row in logged:
            epoch = int(row["epoch"])
            variant_id = f"{config['id']}_epoch_{epoch}"
            variant = evaluate_variant(
                variant_id=variant_id,
                label=f"{config['name']} Epoch {epoch}",
                model_config=config,
                records=records,
                device=device,
                epoch=epoch,
                train_loss=float(row["train_loss"]),
                is_base=False,
            )
            variant_public = {k: v for k, v in variant.items() if k != "scores"}
            variants.append(variant_public)
            model_variant_payloads[variant_id] = variant_public
            variant_scores[variant_id] = variant["scores"]
            model_variants.append(variant_id)
            epoch_metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": float(row["train_loss"]),
                    "validation_loss": variant["validation_loss"],
                    "pairwise_accuracy": variant["metrics"]["pairwise_accuracy"],
                    "hit_at_1": variant["metrics"]["hit_at_1"],
                    "mrr": variant["metrics"]["mrr"],
                    "mean_margin": variant["metrics"]["mean_margin"],
                }
            )

        epoch_variant_ids = [variant_id for variant_id in model_variants if variant_id != base_id]
        best_variant_id = max(
            epoch_variant_ids,
            key=lambda variant_id: (
                model_variant_payloads[variant_id]["metrics"]["mrr"],
                -model_variant_payloads[variant_id]["epoch"],
            ),
        )
        best_epoch = int(model_variant_payloads[best_variant_id]["epoch"])
        models.append(
            {
                "id": config["id"],
                "name": config["name"],
                "model_name": config["model_name"],
                "embedding_dim": config["embedding_dim"],
                "output_dir": str(config["output_dir"]),
                "best_epoch": best_epoch,
                "base_variant": base_id,
                "best_variant": best_variant_id,
                "variant_ids": model_variants,
                "epoch_metrics": epoch_metrics,
            }
        )

    groups = build_group_payload(records, variant_scores, assets_dir=assets_dir)
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "source": {
            "project": str(PROJECT_ROOT),
            "data": str(args.pairs_dir),
        },
        "models": models,
        "variants": variants,
        "groups": groups,
        "ai_comparisons": build_ai_comparisons(groups, variant_scores),
    }
    (args.web_root / "report-data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"models": models, "num_variants": len(variants)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
