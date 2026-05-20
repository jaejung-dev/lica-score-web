from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path("/home/ubuntu/ml-platform/other-projects/lica-score")
WEB_ROOT = Path("/home/ubuntu/lica-score-web")
BASELINE_MODEL_IDS = {"clip_vit_l14_openai", "hpsv2_1"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_groups(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        group = groups.setdefault(
            record["group_id"],
            {
                "prompt": record["condition"]["text_prompt"],
                "candidates": {},
            },
        )
        for side in ("positive", "negative"):
            candidate = record[side]
            group["candidates"][candidate["source"]] = candidate
    return groups


def rank_metrics(records: list[dict[str, Any]], scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    pair_correct = 0
    margins: list[float] = []
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"records": [], "candidates": {}})
    per_task = defaultdict(lambda: {"pair_correct": 0, "pair_total": 0, "ranks": []})
    per_task_ranking = defaultdict(lambda: {"correct": 0, "total": 0, "perfect": 0, "groups": 0})

    for record in records:
        task = record["task"]
        group_id = record["group_id"]
        grouped[(task, group_id)]["records"].append(record)
        grouped[(task, group_id)]["candidates"][record["positive"]["source"]] = record["positive"]
        grouped[(task, group_id)]["candidates"][record["negative"]["source"]] = record["negative"]
        margin = scores[group_id][record["positive"]["source"]] - scores[group_id][record["negative"]["source"]]
        margins.append(margin)
        is_correct = margin > 0.0
        pair_correct += int(is_correct)
        per_task[task]["pair_correct"] += int(is_correct)
        per_task[task]["pair_total"] += 1

    ranks: list[int] = []
    hits_at_1 = 0
    grouped_ranking_correct = 0
    grouped_ranking_total = 0
    group_perfect = 0
    for (task, group_id), payload in grouped.items():
        positive_sources = {record["positive"]["source"] for record in payload["records"]}
        scored = sorted(scores[group_id].items(), key=lambda item: item[1], reverse=True)
        positive_ranks = [
            rank for rank, (source, _) in enumerate(scored, start=1) if source in positive_sources
        ]
        if not positive_ranks:
            continue
        best_rank = min(positive_ranks)
        ranks.append(best_rank)
        per_task[task]["ranks"].append(best_rank)
        hits_at_1 += int(best_rank == 1)
        score_map = dict(scored)
        local_correct = 0
        local_total = 0
        for record in payload["records"]:
            for positive_source in positive_sources:
                negative_sources = set(payload["candidates"]) - positive_sources
                for negative_source in negative_sources:
                    correct = score_map[positive_source] > score_map[negative_source]
                    local_correct += int(correct)
                    local_total += 1
        grouped_ranking_correct += local_correct
        grouped_ranking_total += local_total
        group_perfect += int(local_correct == local_total and local_total > 0)
        per_task_ranking[task]["correct"] += local_correct
        per_task_ranking[task]["total"] += local_total
        per_task_ranking[task]["perfect"] += int(local_correct == local_total and local_total > 0)
        per_task_ranking[task]["groups"] += 1

    num_pairs = len(records)
    num_groups = len(ranks)
    by_task = {}
    for task, stats in per_task.items():
        task_rank_stats = per_task_ranking[task]
        task_ranks = stats["ranks"]
        by_task[task] = {
            "pairwise_accuracy": stats["pair_correct"] / stats["pair_total"] if stats["pair_total"] else 0.0,
            "preference_accuracy": stats["pair_correct"] / stats["pair_total"] if stats["pair_total"] else 0.0,
            "ranking_accuracy": task_rank_stats["correct"] / task_rank_stats["total"] if task_rank_stats["total"] else 0.0,
            "group_perfect_accuracy": task_rank_stats["perfect"] / task_rank_stats["groups"] if task_rank_stats["groups"] else 0.0,
            "hit_at_1": float(np.mean([rank == 1 for rank in task_ranks])) if task_ranks else 0.0,
            "mrr": float(np.mean([1.0 / rank for rank in task_ranks])) if task_ranks else 0.0,
            "mean_positive_rank": float(np.mean(task_ranks)) if task_ranks else 0.0,
            "num_pairs": stats["pair_total"],
            "num_groups": len(task_ranks),
        }

    return {
        "pairwise_accuracy": pair_correct / num_pairs if num_pairs else 0.0,
        "preference_accuracy": pair_correct / num_pairs if num_pairs else 0.0,
        "ranking_accuracy": grouped_ranking_correct / grouped_ranking_total if grouped_ranking_total else 0.0,
        "group_perfect_accuracy": group_perfect / num_groups if num_groups else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "hit_at_1": hits_at_1 / num_groups if num_groups else 0.0,
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0,
        "mean_positive_rank": float(np.mean(ranks)) if ranks else 0.0,
        "num_pairs": num_pairs,
        "num_groups": num_groups,
        "by_task": by_task,
    }


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
    return {
        "accuracy": correct / total if total else 0.0,
        "confusion": {
            "true_gt_pred_gt": correct,
            "true_gt_pred_ai": total - correct,
            "total": total,
        },
        "by_source": {
            source: {
                "correct": vals["correct"],
                "total": vals["total"],
                "accuracy": vals["correct"] / vals["total"] if vals["total"] else 0.0,
            }
            for source, vals in sorted(by_source.items())
        },
    }


def score_clip(groups: dict[str, dict[str, Any]], device: torch.device) -> dict[str, dict[str, float]]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    scores: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for group_id, group in tqdm(groups.items(), desc="CLIP ViT-L/14"):
            sources = sorted(group["candidates"])
            images = [
                preprocess(Image.open(group["candidates"][source]["render_path"]).convert("RGB"))
                for source in sources
            ]
            image_tensor = torch.stack(images).to(device)
            text_tensor = tokenizer([group["prompt"]]).to(device)
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tensor)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            values = (image_features @ text_features.T).squeeze(-1).float().cpu().tolist()
            scores[group_id] = {source: float(value) for source, value in zip(sources, values)}
    return scores


def score_hpsv2(groups: dict[str, dict[str, Any]], device: torch.device) -> dict[str, dict[str, float]]:
    import huggingface_hub
    from hpsv2.img_score import initialize_model, model_dict
    from hpsv2.src.open_clip import get_tokenizer
    from hpsv2.utils import hps_version_map

    initialize_model()
    model = model_dict["model"]
    preprocess = model_dict["preprocess_val"]
    checkpoint_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer("ViT-H-14")
    scores: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for group_id, group in tqdm(groups.items(), desc="HPSv2.1"):
            sources = sorted(group["candidates"])
            images = [
                preprocess(Image.open(group["candidates"][source]["render_path"]).convert("RGB"))
                for source in sources
            ]
            image_tensor = torch.stack(images).to(device=device, non_blocking=True)
            text_tensor = tokenizer([group["prompt"]]).to(device=device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(image_tensor, text_tensor)
                values = (outputs["image_features"] @ outputs["text_features"].T).squeeze(-1)
            scores[group_id] = {
                source: float(value) for source, value in zip(sources, values.float().cpu().tolist())
            }
    return scores


def add_variant(
    *,
    data: dict[str, Any],
    records: list[dict[str, Any]],
    model_id: str,
    model_name: str,
    variant_id: str,
    label: str,
    scores: dict[str, dict[str, float]],
) -> None:
    metrics = rank_metrics(records, scores)
    variant = {
        "id": variant_id,
        "label": label,
        "model_id": model_id,
        "epoch": 0,
        "train_loss": None,
        "validation_loss": None,
        "metrics": metrics,
        "summary": gt_ai_summary(records, scores),
    }
    model = {
        "id": model_id,
        "name": model_name,
        "is_baseline": True,
        "base_variant": variant_id,
        "best_variant": variant_id,
        "variant_ids": [variant_id],
        "epoch_metrics": [
            {
                "epoch": 0,
                "train_loss": None,
                "validation_loss": None,
                "pairwise_accuracy": metrics["pairwise_accuracy"],
                "hit_at_1": metrics["hit_at_1"],
                "mrr": metrics["mrr"],
                "mean_margin": metrics["mean_margin"],
            }
        ],
    }
    data["models"].append(model)
    data["variants"].append(variant)
    for group in data["groups"]:
        group_scores = scores[group["group_id"]]
        group.setdefault("winners", {})[variant_id] = max(group_scores.items(), key=lambda item: item[1])[0]
        for entry in group["entries"]:
            if entry["source"] in group_scores:
                entry.setdefault("scores", {})[variant_id] = group_scores[entry["source"]]


def rebuild_ai_comparisons(data: dict[str, Any]) -> None:
    comparisons = []
    pairs = [("gpt-5.2", "claude"), ("claude", "gemini"), ("gpt-5.2", "gemini")]
    variant_ids = [variant["id"] for variant in data["variants"]]
    for group in data["groups"]:
        entries = {entry["source"]: entry for entry in group["entries"]}
        for left, right in pairs:
            if left not in entries or right not in entries:
                continue
            winners = {}
            for variant_id in variant_ids:
                left_score = entries[left]["scores"].get(variant_id)
                right_score = entries[right]["scores"].get(variant_id)
                if left_score is None or right_score is None:
                    continue
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
    data["ai_comparisons"] = comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, default=PROJECT_ROOT / "data/processed/svg_data_v1")
    parser.add_argument("--report-data", type=Path, default=WEB_ROOT / "report-data.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_jsonl(args.pairs_dir / "eval_pairs.jsonl")
    groups = collect_groups(records)
    device = torch.device(args.device)
    data = json.loads(args.report_data.read_text(encoding="utf-8"))

    data["models"] = [model for model in data["models"] if model["id"] not in BASELINE_MODEL_IDS]
    data["variants"] = [
        variant for variant in data["variants"] if variant["model_id"] not in BASELINE_MODEL_IDS
    ]
    baseline_variant_ids = {"clip_vit_l14_openai", "hpsv2_1"}
    for group in data["groups"]:
        for entry in group["entries"]:
            for variant_id in baseline_variant_ids:
                entry.get("scores", {}).pop(variant_id, None)
        for variant_id in baseline_variant_ids:
            group.get("winners", {}).pop(variant_id, None)

    clip_scores = score_clip(groups, device)
    add_variant(
        data=data,
        records=records,
        model_id="clip_vit_l14_openai",
        model_name="CLIP ViT-L/14",
        variant_id="clip_vit_l14_openai",
        label="CLIP ViT-L/14 OpenAI",
        scores=clip_scores,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hps_scores = score_hpsv2(groups, device)
    add_variant(
        data=data,
        records=records,
        model_id="hpsv2_1",
        model_name="HPSv2.1",
        variant_id="hpsv2_1",
        label="HPSv2.1",
        scores=hps_scores,
    )

    rebuild_ai_comparisons(data)
    args.report_data.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for model in data["models"]:
        variant = next(v for v in data["variants"] if v["id"] == model["best_variant"])
        print(
            model["name"],
            "acc",
            round(variant["summary"]["accuracy"], 4),
            "mrr",
            round(variant["metrics"]["mrr"], 4),
            "hit@1",
            round(variant["metrics"]["hit_at_1"], 4),
        )


if __name__ == "__main__":
    main()
