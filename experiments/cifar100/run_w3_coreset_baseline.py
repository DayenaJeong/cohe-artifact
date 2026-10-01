#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order-path", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n-total", type=int, default=50000)
    ap.add_argument("--budgets", type=str, default="0.1,0.5,0.7")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--data-root", type=str, default="user_inputs/cifar100")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    alpha = load_alpha_mix_module()

    order = np.load(args.order_path).astype(np.int64)
    order_n = len(order)
    n_total = int(args.n_total)
    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

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

    rows: list[dict[str, float | int | str]] = []
    manifest = {
        "order_path": str(Path(args.order_path).resolve()),
        "source_strategy": "Coreset-kCenter",
        "budgets": budgets,
        "seeds": seeds,
        "n_total": n_total,
        "order_length": order_n,
        "train_cfg": cfg.__dict__,
        "results": [],
    }

    for budget in budgets:
        k = int(round(budget * n_total))
        if k > order_n:
            raise ValueError(f"Requested K={k} exceeds available coreset order length {order_n}")
        idx = order[:k]
        for seed in seeds:
            alpha.set_seed(seed)
            run_name = f"cifar100_budget{budget:.2f}_K{k}_seed{seed}_Coreset-kCenter"
            safe_name = (
                run_name.replace(" ", "")
                .replace("/", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("=", "")
            )
            acc = alpha.train_and_eval(safe_name, idx, cfg, args.device, save_dir=str(models_dir))
            row = {
                "budget": float(budget),
                "K": int(k),
                "seed": int(seed),
                "strategy": "Coreset-kCenter",
                "acc": float(acc),
            }
            rows.append(row)
            manifest["results"].append(row)
            (out_dir / "coreset_results.json").write_text(json.dumps(manifest, indent=2) + "\n")

    csv_path = out_dir / "coreset_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["budget", "K", "seed", "strategy", "acc"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
