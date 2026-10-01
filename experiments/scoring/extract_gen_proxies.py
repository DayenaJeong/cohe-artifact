import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# CIFAR-100 공식 Fine-to-Coarse 매핑 리스트 (index: fine_id, value: coarse_id)
# 100개 클래스를 20개 슈퍼클래스로 매핑하는 표준 리스트입니다.
FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13
]

def run_granularity_analysis():
    # 1. 경로 설정
    BASE_DIR = "user_inputs/ICML"
    MODEL_PATH = os.path.join(BASE_DIR, "model_Random.pth")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    GEN_HARDNESS_PATH = os.path.join(BASE_DIR, "cifar100_diffusion_scores.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 Using Device: {device}")

    # 2. 모델 로드 (OrderedDict 에러 방지)
    model = models.resnet18(num_classes=100)
    print(f"🔄 Loading weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state_dict, nn.Module):
        model = state_dict
    else:
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # 3. 데이터 로드 (50,000개 학습 데이터)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_set = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=False)

    # 4. 로짓 및 난이도 추출
    print("🚀 Extracting logits and calculating hardness...")
    all_probs = []
    with torch.no_grad():
        for inputs, _ in train_loader:
            outputs = model(inputs.to(device))
            all_probs.append(F.softmax(outputs, dim=1).cpu())

    probs = torch.cat(all_probs, dim=0)

    # (A) Fine Difficulty (100 classes)
    fine_targets = torch.tensor(train_set.targets)
    d_disc_fine = -torch.log(probs[range(len(fine_targets)), fine_targets]).numpy()

    # (B) Coarse Difficulty (20 super-classes)
    # 매핑 행렬 생성 (100 -> 20)
    mapping = np.zeros((100, 20))
    for fine_id, coarse_id in enumerate(FINE_TO_COARSE):
        mapping[fine_id, coarse_id] = 1.0
    mapping_tensor = torch.from_numpy(mapping).float()

    # 슈퍼클래스 확률 계산: P(coarse|x) = sum(P(fine_i|x))
    coarse_probs = torch.matmul(probs, mapping_tensor)
    coarse_targets = torch.tensor([FINE_TO_COARSE[t] for t in train_set.targets])
    d_disc_coarse = -torch.log(coarse_probs[range(len(coarse_targets)), coarse_targets]).numpy()

    # 5. 생성 난이도 로드 및 상관관계 분석
    d_gen = np.load(GEN_HARDNESS_PATH)
    if d_gen.shape[0] != d_disc_fine.shape[0]:
        print(f"⚠️ Warning: Sample count mismatch ({len(d_gen)} vs {len(d_disc_fine)})")
        return

    r_fine, _ = pearsonr(d_gen, d_disc_fine)
    r_coarse, _ = pearsonr(d_gen, d_disc_coarse)

    print("\n" + "=".center(60, "="))
    print("📈 Appendix G: Impact of Label Granularity".center(60))
    print("-".center(60, "-"))
    print(f" Pearson Correlation ($r$) with Generative Hardness:")
    print(f"  - vs. Fine-Disc (100 classes):   {r_fine:.4f}")
    print(f"  - vs. Coarse-Disc (20 classes):  {r_coarse:.4f}")
    print("=".center(60, "="))

if __name__ == "__main__":
    run_granularity_analysis()
