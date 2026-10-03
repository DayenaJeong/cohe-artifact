#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


ALPHA_MIX_PATH = Path(__file__).resolve().parent / 'alpha_mix_baseline.py'


def load_alpha_mix_module():
    spec = importlib.util.spec_from_file_location("alpha_mix_baseline_mod", ALPHA_MIX_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {ALPHA_MIX_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_labels(cifar_root: Path) -> np.ndarray:
    train_path = cifar_root / "cifar-100-python" / "train"
    if not train_path.exists():
        train_path = cifar_root / "train"
    with train_path.open("rb") as f:
        payload = pickle.load(f, encoding="bytes")
    return np.asarray(payload[b"fine_labels"], dtype=np.int64)


def class_quotas(labels: np.ndarray, k_total: int) -> np.ndarray:
    n_classes = int(labels.max()) + 1
    base = k_total // n_classes
    rem = k_total % n_classes
    q = np.full(n_classes, base, dtype=np.int64)
    if rem > 0:
        q[:rem] += 1
    return q


def balanced_random_indices(labels: np.ndarray, k_total: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    quotas = class_quotas(labels, k_total)
    picks = []
    for c, q in enumerate(quotas):
        cls_idx = np.where(labels == c)[0]
        picks.append(rng.choice(cls_idx, size=int(q), replace=False))
    return np.sort(np.concatenate(picks).astype(np.int64))


def balanced_topk_indices(scores: np.ndarray, labels: np.ndarray, k_total: int) -> np.ndarray:
    quotas = class_quotas(labels, k_total)
    picks = []
    for c, q in enumerate(quotas):
        cls_idx = np.where(labels == c)[0]
        cls_scores = scores[cls_idx]
        order = np.argsort(cls_scores)
        picks.append(cls_idx[order[-int(q):]])
    return np.sort(np.concatenate(picks).astype(np.int64))


def subset_metrics(indices: np.ndarray, labels: np.ndarray, dgen: np.ndarray, ddisc: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(labels[indices], minlength=int(labels.max()) + 1).astype(float)
    probs = counts / counts.sum()
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(len(probs)))
    q90_disc = float(np.quantile(ddisc, 0.9))
    q75_gen = float(np.quantile(dgen, 0.75))
    q75_disc = float(np.quantile(ddisc, 0.75))
    sel_gen = dgen[indices]
    sel_disc = ddisc[indices]
    return {
        "n_selected": int(len(indices)),
        "classes_present": int((counts > 0).sum()),
        "class_entropy_norm": entropy,
        "avg_disc_ce": float(sel_disc.mean()),
        "avg_gen": float(sel_gen.mean()),
        "frac_top10_disc": float((sel_disc >= q90_disc).mean()),
        "frac_double_hard_q75": float(((sel_gen >= q75_gen) & (sel_disc >= q75_disc)).mean()),
        "max_class_share": float(probs.max()),
        "l1_to_uniform": float(np.abs(probs - 1.0 / len(probs)).sum()),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["balanced", "unbalanced"], required=True)
    ap.add_argument("--strategy", choices=["Random", "Gen-Hard", "Disc-Hard"], required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dgen-path", type=str, default="user_inputs/ICML/train_d_gen.npy")
    ap.add_argument("--ddisc-path", type=str, default="user_inputs/ICML/train_d_disc.npy")
    ap.add_argument("--data-root", type=str, default="user_inputs/cifar100")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", type=str, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    alpha = load_alpha_mix_module()

    dgen = np.load(args.dgen_path).astype(np.float64)
    ddisc = np.load(args.ddisc_path).astype(np.float64)
    labels = load_labels(Path(args.data_root))
    n = len(labels)
    k = int(round(args.budget * n))

    if args.mode == "balanced":
        if args.strategy == "Random":
            idx = balanced_random_indices(labels, k, args.seed)
            selection_seed = int(args.seed)
        elif args.strategy == "Gen-Hard":
            idx = balanced_topk_indices(dgen, labels, k)
            selection_seed = -1
        else:
            idx = balanced_topk_indices(ddisc, labels, k)
            selection_seed = -1
    else:
        rng = np.random.RandomState(args.seed)
        if args.strategy == "Random":
            idx = alpha.random_indices(n, k, rng)
            selection_seed = int(args.seed)
        elif args.strategy == "Gen-Hard":
            idx = alpha.topk_indices(dgen, k)
            selection_seed = -1
        else:
            idx = alpha.topk_indices(ddisc, k)
            selection_seed = -1
        idx = np.sort(idx.astype(np.int64))

    cfg = alpha.TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        data_root=args.data_root,
        model_name="resnet18",
        num_classes=100,
    )

    alpha.set_seed(args.seed)
    tag = f"{args.mode}_{args.strategy}".replace(" ", "")
    run_name = f"cifar100_budget{args.budget:.2f}_K{k}_seed{args.seed}_{tag}"
    safe_name = run_name.replace("/", "_").replace("(", "").replace(")", "").replace("=", "")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    acc = alpha.train_and_eval(safe_name, idx, cfg, args.device, save_dir=str(models_dir))
    diag = subset_metrics(idx, labels, dgen, ddisc)
    payload = {
        "mode": args.mode,
        "strategy": args.strategy,
        "budget": float(args.budget),
        "K": int(k),
        "train_seed": int(args.seed),
        "selection_seed": int(selection_seed),
        "score_paths": {
            "dgen_path": str(Path(args.dgen_path).resolve()),
            "ddisc_path": str(Path(args.ddisc_path).resolve()),
        },
        "data_root": str(Path(args.data_root).resolve()),
        "selected_indices_path": str((out_dir / f"{safe_name}.indices.npy").resolve()),
        "accuracy": float(acc),
        "diagnostics": diag,
        "train_cfg": cfg.__dict__,
    }
    np.save(out_dir / f"{safe_name}.indices.npy", idx.astype(np.int64))
    (out_dir / f"{safe_name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
