#!/usr/bin/env python3
"""DINOv2 representation-level proxy stress test for COHE.

This script scores CIFAR-100 train examples with a label-free representation
density proxy. For L2-normalized DINOv2 features, distance(u, v) = 1 - cos(u, v),
and the score is the mean distance to the k nearest neighbors excluding self.
Higher score means more isolated in DINOv2 feature space. The script then audits
the proxy against sample-indexed discriminative target arrays using dependence
and held-out predictive validity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "dinov2_representation_stress"
DEFAULT_DATA_ROOT = Path("user_inputs/cifar100")
ICML_ROOT = Path("user_inputs/ICML")


@dataclass(frozen=True)
class TargetSpec:
    name: str
    path: Path
    required: bool = False


def default_targets() -> list[TargetSpec]:
    ttal = ICML_ROOT / "rebuttal_2026" / "ttal_full_response"
    return [
        TargetSpec("CE loss", ICML_ROOT / "outputs" / "cifar100" / "ce_train.npy", required=True),
        TargetSpec("Margin hardness", ttal / "results" / "cifar100_disc_metrics_seed0" / "margin_hardness.npy", required=True),
        TargetSpec("GradNorm", ICML_ROOT / "outputs" / "cifar100" / "gradnorm_train.npy", required=True),
        TargetSpec("First-learning epoch", ICML_ROOT / "rebuttal_2026" / "exp_training_dynamics" / "first_learning_epoch.npy"),
        TargetSpec("Forgetting count", ICML_ROOT / "rebuttal_2026" / "exp_training_dynamics" / "forgetting_counts.npy"),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ci95(vals: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(vals), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        val = float(arr[0])
        return val, val
    mean = float(arr.mean())
    half = 1.96 * float(arr.std(ddof=1)) / math.sqrt(arr.size)
    return mean - half, mean + half


def normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def write_not_run(out_dir: Path, reason: str, details: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    message = [
        "# DINOv2 Representation Stress Test: Not Run",
        "",
        reason.strip(),
        "",
        "No DINOv2 proxy, dependence metrics, or predictive-validity values were produced.",
    ]
    if details:
        message.extend(["", "## Details", "", "```text", details.strip(), "```"])
    (out_dir / "NOT_RUN.md").write_text("\n".join(message) + "\n")


def parse_target_override(spec: str) -> TargetSpec:
    if "=" not in spec:
        raise ValueError(f"Target override must have form name=path, got: {spec}")
    name, path = spec.split("=", 1)
    return TargetSpec(name.strip(), Path(path).expanduser().resolve(), required=True)


def load_targets(target_specs: list[TargetSpec], expected_n: int) -> tuple[dict[str, np.ndarray], list[str]]:
    targets: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    missing_required: list[str] = []
    for spec in target_specs:
        if not spec.path.exists():
            skipped.append(f"{spec.name}: missing {spec.path}")
            if spec.required:
                missing_required.append(f"{spec.name}: {spec.path}")
            continue
        arr = np.load(spec.path).reshape(-1).astype(np.float64)
        if len(arr) != expected_n:
            skipped.append(f"{spec.name}: length {len(arr)} != {expected_n}")
            if spec.required:
                missing_required.append(f"{spec.name}: length mismatch")
            continue
        targets[spec.name] = arr
    if missing_required:
        raise FileNotFoundError("Required target arrays missing or invalid:\n" + "\n".join(missing_required))
    return targets, skipped


def load_hf_model(model_name: str, device: str):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return "hf", processor, model


def load_torchhub_model(model_name: str, device: str):
    import torch

    hub_name = "dinov2_vits14" if model_name == "facebook/dinov2-small" else model_name
    model = torch.hub.load("facebookresearch/dinov2", hub_name)
    model = model.to(device)
    model.eval()
    return "tensor", None, model


def load_timm_model(device: str):
    import timm

    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
    model = model.to(device)
    model.eval()
    return "tensor", None, model


def load_model(model_source: str, model_name: str, device: str):
    errors: list[str] = []
    loaders = []
    if model_source in {"auto", "hf"}:
        loaders.append(("HuggingFace transformers", lambda: load_hf_model(model_name, device)))
    if model_source in {"auto", "torchhub"}:
        loaders.append(("torch.hub", lambda: load_torchhub_model(model_name, device)))
    if model_source in {"auto", "timm"}:
        loaders.append(("timm", lambda: load_timm_model(device)))

    for label, loader in loaders:
        try:
            source, processor, model = loader()
            return source, processor, model, label
        except Exception as exc:  # noqa: BLE001 - all loader failures are reported.
            errors.append(f"[{label}] {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load a DINOv2 model.\n" + "\n".join(errors))


def make_cifar_loader(data_root: Path, batch_size: int, num_workers: int, mode: str, image_size: int):
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.transforms import InterpolationMode

    if mode == "hf":
        transform = None
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    base = torchvision.datasets.CIFAR100(root=str(data_root), train=True, download=False, transform=transform)

    class IndexedDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(base)

        def __getitem__(self, idx: int):
            image, label = base[idx]
            return idx, image, label

    if mode == "hf":

        def collate(batch):
            indices, images, labels = zip(*batch)
            return list(indices), list(images), list(labels)

    else:

        def collate(batch):
            indices, images, labels = zip(*batch)
            return list(indices), torch.stack(list(images), dim=0), list(labels)

    loader = torch.utils.data.DataLoader(
        IndexedDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    return loader, len(base)


def tensor_model_features(model, inputs):
    outputs = None
    if hasattr(model, "forward_features"):
        outputs = model.forward_features(inputs)
    else:
        outputs = model(inputs)
    if isinstance(outputs, dict):
        for key in ("x_norm_clstoken", "cls_token", "pooled", "features"):
            if key in outputs:
                return outputs[key]
        if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"][:, 0]
        raise KeyError(f"Could not find CLS-like feature in model output keys: {sorted(outputs.keys())}")
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if getattr(outputs, "ndim", None) == 3:
        return outputs[:, 0]
    return outputs


def extract_features(
    feature_path: Path,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    device_arg: str,
    model_source: str,
    model_name: str,
    image_size: int,
) -> np.ndarray:
    if feature_path.exists():
        return normalize_rows(np.load(feature_path))

    import torch

    device = device_arg
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    source, processor, model, loaded_from = load_model(model_source, model_name, device)
    loader, n = make_cifar_loader(data_root, batch_size, num_workers, source, image_size)
    print(f"Loaded DINOv2 via {loaded_from}; extracting {n} CIFAR-100 train features on {device}.", flush=True)

    features = np.empty((n, 0), dtype=np.float32)
    offset_initialized = False
    with torch.inference_mode():
        for batch_i, (indices, images, _labels) in enumerate(loader):
            if source == "hf":
                inputs = processor(images=images, return_tensors="pt")
                inputs = {key: val.to(device, non_blocking=True) for key, val in inputs.items()}
                outputs = model(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    feats = outputs.last_hidden_state[:, 0]
                elif hasattr(outputs, "pooler_output"):
                    feats = outputs.pooler_output
                else:
                    raise RuntimeError("HuggingFace DINOv2 output has neither last_hidden_state nor pooler_output.")
            else:
                inputs = images.to(device, non_blocking=True)
                feats = tensor_model_features(model, inputs)
            feats_np = feats.detach().float().cpu().numpy()
            if not offset_initialized:
                features = np.empty((n, feats_np.shape[1]), dtype=np.float32)
                offset_initialized = True
            features[np.asarray(indices, dtype=np.int64)] = feats_np
            if (batch_i + 1) % 20 == 0:
                print(f"  extracted {(batch_i + 1) * batch_size} / {n}", flush=True)

    features = normalize_rows(features)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, features)
    return features


def compute_isolation(features: np.ndarray, k: int) -> np.ndarray:
    features = normalize_rows(features)
    if k < 1:
        raise ValueError("k must be positive.")
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(features.shape[1])
        index.add(features.astype(np.float32))
        sim, ind = index.search(features.astype(np.float32), k + 1)
        # Features are L2-normalized, so inner product is cosine similarity.
        # The isolation proxy uses cosine distance, 1 - cosine similarity;
        # higher scores mean farther from neighbors and therefore more isolated.
        distances = 1.0 - sim
        rows = []
        for i in range(len(features)):
            row_dist = distances[i]
            row_ind = ind[i]
            keep = row_ind != i
            rows.append(row_dist[keep][:k].mean())
        return np.asarray(rows, dtype=np.float64)
    except Exception:
        # sklearn's cosine metric is distance(u, v) = 1 - cos(u, v).
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(features)
        distances, indices = nn.kneighbors(features, return_distance=True)
        scores = np.empty(len(features), dtype=np.float64)
        for i in range(len(features)):
            keep = indices[i] != i
            scores[i] = float(distances[i][keep][:k].mean())
        return scores


def dependence_rows(proxy: np.ndarray, targets: dict[str, np.ndarray], k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_name, target in targets.items():
        mask = np.isfinite(proxy) & np.isfinite(target)
        x = proxy[mask]
        y = target[mask]
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)
        rows.append(
            {
                "dataset": "CIFAR-100 train",
                "proxy": f"DINOv2 feature-space isolation (k={k})",
                "target": target_name,
                "n": int(mask.sum()),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_rho": float(spearman_rho),
                "spearman_p": float(spearman_p),
            }
        )
    return rows


def predictive_rows(
    proxy: np.ndarray,
    targets: dict[str, np.ndarray],
    split_seeds: list[int],
    max_train_samples: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_split: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    x_all = proxy.reshape(-1, 1).astype(np.float64)
    model_factories = {
        "LinearRegression": lambda seed: LinearRegression(),
        "HistGBDT": lambda seed: HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            random_state=seed,
        ),
    }

    for target_name, target in targets.items():
        mask = np.isfinite(proxy) & np.isfinite(target)
        x = x_all[mask]
        y = target[mask].astype(np.float64)
        for split_seed in split_seeds:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=split_seed,
                shuffle=True,
            )
            if max_train_samples > 0 and len(x_train) > max_train_samples:
                rng = np.random.default_rng(split_seed)
                keep = rng.choice(len(x_train), size=max_train_samples, replace=False)
                x_train = x_train[keep]
                y_train = y_train[keep]
            for model_name, factory in model_factories.items():
                model = factory(split_seed)
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
                per_split.append(
                    {
                        "dataset": "CIFAR-100 train",
                        "proxy": "DINOv2 feature-space isolation",
                        "target": target_name,
                        "split_seed": split_seed,
                        "model": model_name,
                        "r2": float(r2_score(y_test, pred)),
                    }
                )

        target_rows = [row for row in per_split if row["target"] == target_name]
        model_means: dict[str, np.ndarray] = {}
        for model_name in model_factories:
            model_means[model_name] = np.asarray(
                [float(row["r2"]) for row in target_rows if row["model"] == model_name],
                dtype=float,
            )
        best_model = max(model_means, key=lambda name: float(model_means[name].mean()))
        lin_vals = model_means["LinearRegression"]
        hgb_vals = model_means["HistGBDT"]
        best_vals = model_means[best_model]
        lin_lo, lin_hi = ci95(lin_vals)
        hgb_lo, hgb_hi = ci95(hgb_vals)
        best_lo, best_hi = ci95(best_vals)
        summary.append(
            {
                "dataset": "CIFAR-100 train",
                "proxy": "DINOv2 feature-space isolation",
                "target": target_name,
                "n": int(mask.sum()),
                "linear_r2_mean": float(lin_vals.mean()),
                "linear_r2_ci95_low": lin_lo,
                "linear_r2_ci95_high": lin_hi,
                "histgbdt_r2_mean": float(hgb_vals.mean()),
                "histgbdt_r2_ci95_low": hgb_lo,
                "histgbdt_r2_ci95_high": hgb_hi,
                "best_single_proxy_model": best_model,
                "best_single_proxy_r2_mean": float(best_vals.mean()),
                "best_single_proxy_ci95_low": best_lo,
                "best_single_proxy_ci95_high": best_hi,
                "split_seeds": " ".join(str(seed) for seed in split_seeds),
            }
        )
    return per_split, summary


def cohe_reading(target: str, rho: float, best_r2: float) -> str:
    abs_rho = abs(rho)
    if target in {"CE loss", "Margin hardness"}:
        if abs_rho <= 0.1 and best_r2 <= 0.005:
            return "same low CE/margin surrogate-validity pattern"
        if abs_rho <= 0.2 or best_r2 <= 0.02:
            return "weak representation-level association; not a validated surrogate"
        return "stronger representation-level signal; requires separate COHE audit and transfer evidence"
    if best_r2 <= 0.005 and abs_rho <= 0.1:
        return "low measured dependence"
    return "target-specific relation"


def write_report(
    out_dir: Path,
    dependence: list[dict[str, object]],
    predictive: list[dict[str, object]],
    skipped_targets: list[str],
    feature_path: Path,
    k: int,
) -> None:
    pred_by_target = {str(row["target"]): row for row in predictive}
    lines = [
        "# DINOv2 Representation-Level Stress Test",
        "",
        f"Proxy: mean cosine distance, distance(u, v) = 1 - cos(u, v), to the {k} nearest neighbors in normalized DINOv2 feature space. Higher is more isolated.",
        "",
        "## Outputs",
        "",
        "- `cifar100_dinov2_proxy.csv`",
        "- `cifar100_dinov2_dependence.csv`",
        "- `cifar100_dinov2_predictive_r2.csv`",
        "- `cifar100_dinov2_predictive_r2_per_split.csv`",
        f"- feature cache: `{feature_path.name}`",
        "",
        "## Metrics",
        "",
        "| Target | Pearson r | Spearman rho | Best single-proxy R2 | Best model | COHE reading |",
        "|---|---:|---:|---:|---|---|",
    ]
    for dep in dependence:
        target = str(dep["target"])
        pred = pred_by_target[target]
        rho = float(dep["spearman_rho"])
        best_r2 = float(pred["best_single_proxy_r2_mean"])
        lines.append(
            "| {target} | {r:+.6f} | {rho:+.6f} | {r2:.6f} | {model} | {reading} |".format(
                target=target,
                r=float(dep["pearson_r"]),
                rho=rho,
                r2=best_r2,
                model=pred["best_single_proxy_model"],
                reading=cohe_reading(target, rho, best_r2),
            )
        )
    if skipped_targets:
        lines.extend(["", "## Skipped Optional Targets", ""])
        lines.extend(f"- {item}" for item in skipped_targets)
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-name", default="facebook/dinov2-small")
    parser.add_argument("--model-source", choices=["auto", "hf", "torchhub", "timm"], default="auto")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--split-seeds", default="0,1,2,3,4")
    parser.add_argument("--max-train-samples", type=int, default=40000)
    parser.add_argument("--features-path", type=Path, default=None)
    parser.add_argument("--target", action="append", default=[], help="Additional or replacement target as name=/path/to/array.npy")
    parser.add_argument("--no-default-targets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.features_path or out_dir / "cifar100_dinov2_features.npy"

    target_specs = [] if args.no_default_targets else default_targets()
    target_specs.extend(parse_target_override(spec) for spec in args.target)
    split_seeds = [int(seed) for seed in args.split_seeds.split(",") if seed.strip()]

    try:
        features = extract_features(
            feature_path=feature_path,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_arg=args.device,
            model_source=args.model_source,
            model_name=args.model_name,
            image_size=args.image_size,
        )
    except Exception as exc:  # noqa: BLE001 - writes a truthful no-result report.
        details = traceback.format_exc()
        reason = (
            f"DINOv2 feature extraction failed while loading or running `{args.model_name}`. "
            "This usually means the DINOv2 weights or required model-loading backend are not available locally "
            "and could not be fetched in this environment."
        )
        write_not_run(out_dir, reason, details)
        print(reason, file=sys.stderr)
        print(f"Wrote {out_dir / 'NOT_RUN.md'}", file=sys.stderr)
        return 0

    try:
        targets, skipped_targets = load_targets(target_specs, expected_n=len(features))
    except Exception as exc:  # noqa: BLE001
        details = traceback.format_exc()
        write_not_run(out_dir, f"Target loading failed: {exc}", details)
        print(f"Target loading failed: {exc}", file=sys.stderr)
        return 0

    proxy = compute_isolation(features, k=args.k)
    proxy_csv = out_dir / "cifar100_dinov2_proxy.csv"
    write_csv(
        proxy_csv,
        [
            {
                "sample_id": int(i),
                "dinov2_isolation_k20" if args.k == 20 else f"dinov2_isolation_k{args.k}": float(score),
            }
            for i, score in enumerate(proxy)
        ],
    )
    np.save(out_dir / f"cifar100_dinov2_isolation_k{args.k}.npy", proxy.astype(np.float32))

    dep = dependence_rows(proxy, targets, k=args.k)
    per_split, pred = predictive_rows(proxy, targets, split_seeds, args.max_train_samples)
    write_csv(out_dir / "cifar100_dinov2_dependence.csv", dep)
    write_csv(out_dir / "cifar100_dinov2_predictive_r2_per_split.csv", per_split)
    write_csv(out_dir / "cifar100_dinov2_predictive_r2.csv", pred)
    write_report(out_dir, dep, pred, skipped_targets, feature_path, args.k)

    manifest = {
        "dataset": "CIFAR-100 train",
        "sample_ordering": "torchvision.datasets.CIFAR100(train=True), original order",
        "model_name": args.model_name,
        "model_source": args.model_source,
        "feature_path": str(feature_path),
        "k": args.k,
        "score_definition": "mean_{j in kNN(i), j != i} [1 - cos(z_i, z_j)] using L2-normalized DINOv2 features",
        "score_direction": "higher means more isolated in representation space",
        "split_seeds": split_seeds,
        "outputs": {
            "proxy_csv": str(proxy_csv),
            "dependence_csv": str(out_dir / "cifar100_dinov2_dependence.csv"),
            "predictive_r2_csv": str(out_dir / "cifar100_dinov2_predictive_r2.csv"),
            "report": str(out_dir / "REPORT.md"),
        },
        "skipped_targets": skipped_targets,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
