import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import timm


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic은 약간 느릴 수 있음. 필요하면 True로.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_scores(gen_path: str, disc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    d_gen = np.load(gen_path)
    d_disc = np.load(disc_path)
    assert len(d_gen) == len(d_disc), f"Length mismatch: {len(d_gen)} vs {len(d_disc)}"
    return d_gen.astype(np.float64), d_disc.astype(np.float64)


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = x.mean()
    sd = x.std()
    return (x - mu) / (sd + eps)


def topk_indices(x: np.ndarray, k: int) -> np.ndarray:
    # 큰 값 top-k
    return np.argsort(x)[-k:]


def random_indices(n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    return rng.choice(np.arange(n), size=k, replace=False)


def ohs_xor_indices(d_gen: np.ndarray, d_disc: np.ndarray, k: int, percentile: float, rng: np.random.RandomState) -> np.ndarray:
    # XOR: (gen hard & disc easy) U (disc hard & gen easy)
    tau_gen = np.percentile(d_gen, percentile)
    tau_disc = np.percentile(d_disc, percentile)
    hard_gen = d_gen > tau_gen
    hard_disc = d_disc > tau_disc
    xor_mask = np.logical_xor(hard_gen, hard_disc)
    pool = np.where(xor_mask)[0]
    if len(pool) < k:
        # pool이 작으면 fallback: pool 전부 + 나머지 random
        remaining = k - len(pool)
        rest_pool = np.setdiff1d(np.arange(len(d_gen)), pool)
        extra = rng.choice(rest_pool, size=remaining, replace=False)
        return np.concatenate([pool, extra])
    return rng.choice(pool, size=k, replace=False)


# -------------------------
# Training / Eval
# -------------------------
@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 4
    data_root: str = "./data"
    model_name: str = "resnet18"
    num_classes: int = 100


def build_loaders(selected_indices: np.ndarray, cfg: TrainConfig):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    full_train_set = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=False, download=True, transform=test_transform
    )

    train_subset = Subset(full_train_set, selected_indices.tolist())
    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=100, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, test_loader


def train_and_eval(run_name: str, selected_indices: np.ndarray, cfg: TrainConfig, device: str, save_dir: str) -> float:
    print(f"\n🚀 Starting Experiment: {run_name} | K={len(selected_indices)}")

    train_loader, test_loader = build_loaders(selected_indices, cfg)

    model = timm.create_model(cfg.model_name, pretrained=False, num_classes=cfg.num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % max(1, cfg.epochs // 10) == 0:
            print(f"  Epoch [{epoch+1:3d}/{cfg.epochs}] | loss={train_loss/len(train_loader):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    acc = 100.0 * correct / total
    print(f"✅ Final Test Accuracy for {run_name}: {acc:.2f}%")

    # save model
    ensure_dir(save_dir)
    ckpt_path = os.path.join(save_dir, f"{run_name}.pth")
    torch.save(model.state_dict(), ckpt_path)
    return acc


# -------------------------
# Selection strategies
# -------------------------
def build_selection_indices(
    d_gen: np.ndarray,
    d_disc: np.ndarray,
    K: int,
    seed: int,
    alphas: List[float],
    percentile: float
) -> Dict[str, np.ndarray]:
    n = len(d_gen)
    rng = np.random.RandomState(seed)

    # raw top-k baselines
    out = {}
    out["Random"] = random_indices(n, K, rng)
    out["Gen-Hard"] = topk_indices(d_gen, K)
    out["Disc-Hard"] = topk_indices(d_disc, K)

    # OHS-XOR
    out[f"OHS-XOR(p{int(percentile)})"] = ohs_xor_indices(d_gen, d_disc, K, percentile, rng)

    # alpha-mix: mix-hard = topK of alpha*z(gen) + (1-alpha)*z(disc)
    zg = zscore(d_gen)
    zd = zscore(d_disc)
    for a in alphas:
        mix = a * zg + (1.0 - a) * zd
        out[f"Mix-Hard(a={a:.2f})"] = topk_indices(mix, K)

    return out


# -------------------------
# Reporting
# -------------------------
def save_results_json(path: str, payload: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved] {path}")


def to_latex_table(results: List[dict], budgets: List[float], seeds: List[int], alphas: List[float], percentile: float) -> str:
    """
    results: list of rows with keys: budget, seed, strategy, acc
    """
    # aggregate mean/std per (budget, strategy)
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in results:
        key = (r["budget"], r["strategy"])
        bucket[key].append(r["acc"])

    # strategies order
    strat_order = ["Random", "Gen-Hard", "Disc-Hard", f"OHS-XOR(p{int(percentile)})"] + [f"Mix-Hard(a={a:.2f})" for a in alphas]

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{\\textbf{\\(\\alpha\\)-mix scalarization baseline on CIFAR-100.} Test accuracy (\\%) of ResNet-18 trained on selected subsets. "
                 "We report mean$\\pm$std over seeds for each budget. Mix-Hard ranks samples by "
                 "$D_{\\text{mix}} = \\alpha\\,\\mathrm{z}(D_{\\text{gen}}) + (1-\\alpha)\\,\\mathrm{z}(D_{\\text{disc}})$.}")
    lines.append("\\label{tab:alpha_mix}")
    lines.append("\\begin{small}")
    lines.append("\\begin{sc}")

    # header
    header = "Budget & " + " & ".join([s.replace("_", "\\_") for s in strat_order]) + " \\\\"
    lines.append("\\begin{tabular}{l" + "c"*len(strat_order) + "}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")

    for b in budgets:
        row = [f"{b:.1f}"]
        # compute best to bold
        vals = []
        for s in strat_order:
            accs = bucket.get((b, s), [])
            if len(accs) == 0:
                vals.append(None)
            else:
                m = float(np.mean(accs))
                sd = float(np.std(accs))
                vals.append((m, sd))

        best_m = max([v[0] for v in vals if v is not None], default=None)

        for v in vals:
            if v is None:
                row.append("--")
            else:
                m, sd = v
                cell = f"{m:.2f} $\\pm$ {sd:.2f}"
                if best_m is not None and abs(m - best_m) < 1e-9:
                    cell = "\\textbf{" + cell + "}"
                row.append(cell)

        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{sc}")
    lines.append("\\end{small}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", type=str, default="train_d_gen.npy", help="Path to D_gen .npy")
    p.add_argument("--disc", type=str, default="train_d_disc.npy", help="Path to D_disc .npy")
    p.add_argument("--out_dir", type=str, default="outputs_alpha_mix", help="Directory to save results/models")

    # sweep
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    p.add_argument("--budgets", type=str, default="0.1,0.5,0.7", help="Comma-separated budgets (fraction of 50k)")
    p.add_argument("--K", type=int, default=-1, help="If set (>0), override budget and use fixed K")

    p.add_argument("--alphas", type=str, default="0.00,0.25,0.50,0.75,1.00", help="Comma-separated alpha values")
    p.add_argument("--ohs_percentile", type=float, default=75.0, help="Percentile threshold for OHS-XOR")

    # train
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--data_root", type=str, default="./data")
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    budgets = [float(b.strip()) for b in args.budgets.split(",") if b.strip() != ""]
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip() != ""]
    percentile = float(args.ohs_percentile)

    d_gen, d_disc = load_scores(args.gen, args.disc)
    n = len(d_gen)

    cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        data_root=args.data_root,
        model_name="resnet18",
        num_classes=100
    )

    ensure_dir(args.out_dir)
    models_dir = os.path.join(args.out_dir, "models")
    ensure_dir(models_dir)

    all_rows = []
    summary = {
        "paths": {"gen": args.gen, "disc": args.disc},
        "n_total": n,
        "seeds": seeds,
        "budgets": budgets,
        "K_override": args.K,
        "alphas": alphas,
        "ohs_percentile": percentile,
        "train_cfg": cfg.__dict__,
        "results": []
    }

    for budget in budgets:
        if args.K > 0:
            K = int(args.K)
        else:
            K = int(round(budget * n))
        K = max(1, min(K, n))

        for seed in seeds:
            set_seed(seed)

            # selection indices
            sel = build_selection_indices(d_gen, d_disc, K=K, seed=seed, alphas=alphas, percentile=percentile)

            for strategy, idx in sel.items():
                run_name = f"cifar100_budget{budget:.2f}_K{K}_seed{seed}_{strategy}"
                # 파일명 안전 처리
                safe_name = run_name.replace(" ", "").replace("/", "_").replace("(", "").replace(")", "").replace("=", "")
                acc = train_and_eval(safe_name, idx, cfg, device, save_dir=models_dir)

                row = {
                    "budget": float(budget),
                    "K": int(K),
                    "seed": int(seed),
                    "strategy": strategy,
                    "acc": float(acc)
                }
                all_rows.append(row)
                summary["results"].append(row)

                # 중간 저장(실패해도 남게)
                save_results_json(os.path.join(args.out_dir, "alpha_mix_results.json"), summary)

    # final artifacts
    latex = to_latex_table(all_rows, budgets=budgets, seeds=seeds, alphas=alphas, percentile=percentile)
    latex_path = os.path.join(args.out_dir, "alpha_mix_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"[Saved] {latex_path}")

    print("\n" + "=" * 60)
    print("Done. Summary saved to:")
    print(f"  - {os.path.join(args.out_dir, 'alpha_mix_results.json')}")
    print(f"  - {latex_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
