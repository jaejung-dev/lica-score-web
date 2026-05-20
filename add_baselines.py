from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from cache_config import configure_external_caches


configure_external_caches()

PROJECT_ROOT = Path("/home/ubuntu/ml-platform/other-projects/lica-score")
WEB_ROOT = Path("/home/ubuntu/lica-score-web")

IMSCORE_MODELS = [
    {
        "id": "imscore_hpsv21",
        "name": "HPSv2.1",
        "class": "imscore.hps.model:HPSv2",
        "repo": "RE-N-Y/hpsv21",
    },
    {
        "id": "imscore_pickscore",
        "name": "PickScore",
        "class": "imscore.pickscore.model:PickScorer",
        "repo": "yuvalkirstain/PickScore_v1",
        "constructor": True,
    },
    {
        "id": "imscore_mpsv1",
        "name": "MPS v1",
        "class": "imscore.mps.model:MPS",
        "repo": "RE-N-Y/mpsv1",
    },
    {
        "id": "imscore_imagereward",
        "name": "ImageReward",
        "class": "imscore.imreward.model:ImageReward",
        "repo": "RE-N-Y/ImageReward",
    },
    {
        "id": "imscore_clipscore",
        "name": "CLIPScore",
        "class": "imscore.preference.model:CLIPScore",
        "repo": "RE-N-Y/clipscore-vit-large-patch14",
    },
    {
        "id": "imscore_pickscore_siglip",
        "name": "SigLIP PickScore",
        "class": "imscore.preference.model:SiglipPreferenceScorer",
        "repo": "RE-N-Y/pickscore-siglip",
    },
    {
        "id": "imscore_pickscore_clip",
        "name": "CLIP PickScore",
        "class": "imscore.preference.model:CLIPPreferenceScorer",
        "repo": "RE-N-Y/pickscore-clip",
    },
    {
        "id": "imscore_laion_aesthetic",
        "name": "LAION Aesthetic",
        "class": "imscore.aesthetic.model:LAIONAestheticScorer",
        "repo": "RE-N-Y/laion-aesthetic",
    },
    {
        "id": "imscore_shadow_aesthetic",
        "name": "Shadow Aesthetic",
        "class": "imscore.aesthetic.model:ShadowAesthetic",
        "repo": "RE-N-Y/aesthetic-shadow-v2",
    },
    {
        "id": "imscore_vqascore",
        "name": "VQAScore",
        "class": "imscore.vqascore.model:VQAScore",
        "repo": "RE-N-Y/clip-t5-xxl",
        "batch_size": 1,
    },
    {
        "id": "imscore_evalmuse",
        "name": "EvalMuse",
        "class": "imscore.evalmuse.model:EvalMuse",
        "repo": "RE-N-Y/evalmuse",
    },
    {
        "id": "imscore_hpsv3",
        "name": "HPSv3",
        "class": "imscore.hpsv3.model:HPSv3",
        "repo": "RE-N-Y/hpsv3",
    },
    {
        "id": "imscore_cyclereward_combo",
        "name": "CycleReward Combo",
        "class": "imscore.cyclereward.model:CycleReward",
        "repo": "NagaSaiAbhinay/CycleReward-Combo",
    },
    {
        "id": "imscore_cyclereward_t2i",
        "name": "CycleReward T2I",
        "class": "imscore.cyclereward.model:CycleReward",
        "repo": "NagaSaiAbhinay/CycleReward-T2I",
    },
    {
        "id": "imscore_cyclereward_i2t",
        "name": "CycleReward I2T",
        "class": "imscore.cyclereward.model:CycleReward",
        "repo": "NagaSaiAbhinay/CycleReward-I2T",
    },
]


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


def load_imscore_model(spec: dict[str, Any], device: torch.device) -> Any:
    if spec["id"] == "imscore_mpsv1":
        patch_clip_text_return_dict()
    module_name, class_name = spec["class"].split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    if spec.get("constructor"):
        model = cls(spec["repo"])
    else:
        model = cls.from_pretrained(spec["repo"])
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def patch_clip_text_return_dict() -> None:
    from transformers.models.clip.modeling_clip import CLIPTextTransformer, CLIPVisionTransformer

    def patch_forward(cls: Any) -> None:
        if getattr(cls.forward, "_lica_patched_return_dict", False):
            return

        original_forward = cls.forward

        def forward_without_return_dict(self: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs.pop("return_dict", None)
            return original_forward(self, *args, **kwargs)

        forward_without_return_dict._lica_patched_return_dict = True  # type: ignore[attr-defined]
        cls.forward = forward_without_return_dict

    patch_forward(CLIPTextTransformer)
    patch_forward(CLIPVisionTransformer)


def image_to_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_to_scores(raw: Any, expected: int) -> list[float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        raw = raw[0]
    tensor = torch.as_tensor(raw).detach().float().cpu().reshape(-1)
    if tensor.numel() == 1 and expected > 1:
        return [float(tensor.item())] * expected
    if tensor.numel() < expected:
        raise RuntimeError(f"Expected at least {expected} scores, got shape={tuple(tensor.shape)}")
    return [float(value) for value in tensor[:expected].tolist()]


@torch.no_grad()
def score_pixels(model: Any, pixels: torch.Tensor, prompts: list[str]) -> list[float]:
    raw = model.score(pixels, prompts)
    return tensor_to_scores(raw, pixels.shape[0])


def score_imscore_model(
    spec: dict[str, Any],
    groups: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model = load_imscore_model(spec, device)
    scores: dict[str, dict[str, float]] = {}
    for group_id, group in tqdm(groups.items(), desc=spec["name"]):
        sources = sorted(group["candidates"])
        images = [image_to_tensor(group["candidates"][source]["render_path"]) for source in sources]
        prompts = [group["prompt"]] * len(sources)
        if spec.get("batch_size") == 1:
            values = []
            for image, prompt in zip(images, prompts):
                pixels = image.unsqueeze(0).to(device=device, non_blocking=True)
                values.extend(score_pixels(model, pixels, [prompt]))
        elif len({tuple(image.shape) for image in images}) == 1:
            pixels = torch.stack(images).to(device=device, non_blocking=True)
            values = score_pixels(model, pixels, prompts)
        else:
            values = []
            for image, prompt in zip(images, prompts):
                pixels = image.unsqueeze(0).to(device=device, non_blocking=True)
                values.extend(score_pixels(model, pixels, [prompt]))
        scores[group_id] = {source: float(value) for source, value in zip(sources, values)}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
        "score_display": "imscore_score",
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


def write_report(path: Path, data: dict[str, Any]) -> None:
    data["generated_at"] = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    rebuild_ai_comparisons(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remove_existing_baselines(data: dict[str, Any], selected_model_ids: set[str] | None = None) -> None:
    baseline_model_ids = {model["id"] for model in data["models"] if model.get("is_baseline")}
    if selected_model_ids is not None:
        baseline_model_ids &= selected_model_ids
    baseline_variant_ids = {
        variant["id"] for variant in data["variants"] if variant["model_id"] in baseline_model_ids
    }
    data["models"] = [model for model in data["models"] if model["id"] not in baseline_model_ids]
    data["variants"] = [
        variant for variant in data["variants"] if variant["id"] not in baseline_variant_ids
    ]
    for group in data["groups"]:
        for entry in group["entries"]:
            for variant_id in baseline_variant_ids:
                entry.get("scores", {}).pop(variant_id, None)
        for variant_id in baseline_variant_ids:
            group.get("winners", {}).pop(variant_id, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, default=PROJECT_ROOT / "data/processed/svg_data_v1")
    parser.add_argument("--report-data", type=Path, default=WEB_ROOT / "report-data.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of imscore model ids. Defaults to all configured models.",
    )
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    records = load_jsonl(args.pairs_dir / "eval_pairs.jsonl")
    groups = collect_groups(records)
    device = torch.device(args.device)
    data = json.loads(args.report_data.read_text(encoding="utf-8"))

    specs = IMSCORE_MODELS
    if args.models:
        selected = set(args.models)
        specs = [spec for spec in specs if spec["id"] in selected]
        missing = selected - {spec["id"] for spec in specs}
        if missing:
            raise ValueError(f"Unknown imscore model ids: {sorted(missing)}")
    else:
        selected = None
    remove_existing_baselines(data, selected_model_ids=selected)

    failures = []
    for spec in specs:
        try:
            scores = score_imscore_model(spec, groups, device)
            add_variant(
                data=data,
                records=records,
                model_id=spec["id"],
                model_name=spec["name"],
                variant_id=spec["id"],
                label=spec["name"],
                scores=scores,
            )
            data["baseline_failures"] = failures
            write_report(args.report_data, data)
        except Exception as exc:
            failures.append({"id": spec["id"], "name": spec["name"], "error": repr(exc)})
            print(f"failed {spec['id']}: {exc!r}")
            data["baseline_failures"] = failures
            write_report(args.report_data, data)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not args.keep_going:
                raise

    data["baseline_failures"] = failures
    write_report(args.report_data, data)
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
    if failures:
        print("failures", json.dumps(failures, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
