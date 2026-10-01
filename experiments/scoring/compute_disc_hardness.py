import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision import transforms, models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","cifar100"], default="cifar100")
    ap.add_argument("--split", choices=["train","test"], default="test")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_ce", required=True)
    ap.add_argument("--out_gradnorm", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_ce), exist_ok=True)
    device = torch.device(args.device)

    # [중요] CIFAR-100 표준 Normalization 적용 (ResNet 학습 시 보통 사용함)
    mean = [0.5071, 0.4867, 0.4408] if args.dataset == 'cifar100' else [0.4914, 0.4822, 0.4465]
    std = [0.2675, 0.2565, 0.2761] if args.dataset == 'cifar100' else [0.2023, 0.1994, 0.2010]

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    root = "./data"
    if args.dataset == "cifar10":
        ds = CIFAR10(root=root, train=(args.split=="train"), download=True, transform=tfm)
        num_classes = 10
    else:
        ds = CIFAR100(root=root, train=(args.split=="train"), download=True, transform=tfm)
        num_classes = 100

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f">>> Loading ResNet18 Checkpoint: {args.ckpt}")
    model = models.resnet18(num_classes=num_classes)

    # [수정됨] 체크포인트 로딩 로직 강화 (model_state 대응)
    ckpt = torch.load(args.ckpt, map_location="cpu")

    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']  # 사용자님 케이스
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Remove 'module.' prefix if exists (DataParallel 저장본 대응)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # strict=False로 두면 fc layer 차원 미세 불일치 등을 무시하고 로딩 (안전장치)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[Warning] Strict loading failed, trying strict=False. Error: {e}")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    all_ce = []
    all_gn = []

    for x, y in tqdm(loader, desc="Measuring D_disc"):
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)

        logits = model(x)
        ce = F.cross_entropy(logits, y, reduction="none")
        all_ce.append(ce.detach().cpu().numpy())

        loss = ce.mean()
        model.zero_grad()
        loss.backward()

        gn = x.grad.view(x.size(0), -1).norm(p=2, dim=1)
        all_gn.append(gn.detach().cpu().numpy())

    np.save(args.out_ce, np.concatenate(all_ce))
    np.save(args.out_gradnorm, np.concatenate(all_gn))
    print("[Done] Results saved.")

if __name__ == "__main__":
    main()
