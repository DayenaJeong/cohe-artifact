# import os, argparse
# import numpy as np
# import torch
# from tqdm import tqdm
# from torchvision.datasets import CIFAR100, CIFAR10
# from torchvision import transforms
# from diffusers import DDPMPipeline, DDIMScheduler

# def set_seed(seed: int):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# @torch.no_grad()
# def compute_ddim_hardness(
#     pipe: DDPMPipeline,
#     images: torch.Tensor,
#     num_t_samples: int = 10, # t 샘플 개수 늘림 (더 안정적)
#     seed: int = 0
# ) -> torch.Tensor:
#     device = images.device
#     b = images.shape[0]
#     unet = pipe.unet
#     scheduler = pipe.scheduler
#     num_train_timesteps = scheduler.config.num_train_timesteps

#     # [수정] 랜덤 대신 균등한 간격으로 t를 선정 (Fair Comparison)
#     # 예: 0, 100, 200, ... 900
#     t_seq = np.linspace(0, num_train_timesteps - 1, num_t_samples, dtype=int)
#     timesteps = torch.tensor(t_seq, device=device).long() # [num_t_samples]

#     scores = torch.zeros((b,), device=device)

#     # 각 t에 대해 배치를 반복 (메모리 효율 및 공정성)
#     for t in timesteps:
#         # t를 배치 크기만큼 확장
#         t_batch = t.repeat(b)

#         noise = torch.randn_like(images) # [B, 3, 32, 32]
#         x_t = scheduler.add_noise(images, noise, t_batch)

#         # DDIM/DDPM UNet Inference
#         eps_hat = unet(x_t, t_batch).sample

#         # MSE (Per Image)
#         mse = (noise - eps_hat).pow(2).mean(dim=(1,2,3))
#         scores += mse

#     scores = scores / float(num_t_samples)
#     return scores

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset", choices=["cifar10","cifar100"], default="cifar100")
#     ap.add_argument("--split", choices=["train","test"], default="test") # test로 기본값 변경
#     ap.add_argument("--out", required=True)
#     ap.add_argument("--batch_size", type=int, default=128)
#     ap.add_argument("--t_samples", type=int, default=10) # 10번 정도 찍어야 노이즈 분산이 줄어듦
#     ap.add_argument("--device", default="cuda")
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--model_id", default="google/ddpm-cifar10-32")
#     ap.add_argument("--ddim_steps", type=int, default=50)
#     args = ap.parse_args()

#     set_seed(args.seed)
#     os.makedirs(os.path.dirname(args.out), exist_ok=True)
#     device = torch.device(args.device)

#     # Diffusion용 Transform: [-1, 1]
#     tfm = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x * 2.0 - 1.0),
#     ])

#     root = "./data"
#     if args.dataset == "cifar10":
#         ds = CIFAR10(root=root, train=(args.split=="train"), download=True, transform=tfm)
#     else:
#         ds = CIFAR100(root=root, train=(args.split=="train"), download=True, transform=tfm)

#     loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

#     print(f">>> Loading Diffusion Model: {args.model_id}")
#     pipe = DDPMPipeline.from_pretrained(args.model_id)
#     pipe = pipe.to(device)

#     # DDIM Scheduler 교체
#     pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
#     pipe.scheduler.set_timesteps(args.ddim_steps)

#     all_scores = []
#     print(f">>> Computing Hardness with {args.t_samples} uniform timesteps...")

#     for x, _ in tqdm(loader):
#         x = x.to(device)
#         scores = compute_ddim_hardness(pipe, x, num_t_samples=args.t_samples)
#         all_scores.append(scores.detach().cpu().numpy())

#     all_scores = np.concatenate(all_scores, axis=0)
#     np.save(args.out, all_scores)
#     print(f"[Done] Saved to {args.out}")

# if __name__ == "__main__":
#     main()

import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision import transforms
from diffusers import DDPMPipeline, DDIMScheduler


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_openclip(device, model_name="ViT-B-32", pretrained="openai"):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=str(device),
    )
    model.eval()

    clip_mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073], device=device
    ).view(1,3,1,1)
    clip_std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711], device=device
    ).view(1,3,1,1)
    return model, clip_mean, clip_std


def predict_x0_from_xt_eps(scheduler, x_t, eps_hat, t_batch):
    ac = scheduler.alphas_cumprod.to(x_t.device)
    a_t = ac[t_batch].view(-1,1,1,1)
    x0_hat = (x_t - torch.sqrt(1-a_t)*eps_hat) / (torch.sqrt(a_t)+1e-12)
    return x0_hat


def to_clip_input(x, clip_mean, clip_std, out_size=224):
    x01 = (x + 1) * 0.5
    x01 = x01.clamp(0,1)
    if x01.shape[-1] != out_size:
        x01 = F.interpolate(x01, size=(out_size,out_size),
                            mode="bilinear", align_corners=False)
    return (x01 - clip_mean) / clip_std


# -------------------------
# Core computation
# -------------------------
@torch.no_grad()
def compute_ddim_hardness_and_clip(
    pipe, clip_model, clip_mean, clip_std,
    images, t_list
):
    """
    Returns:
      mse_dict[t]  : noise MSE at fixed timestep t
      clip_dict[t] : CLIP(x, x0_hat) distance at fixed timestep t
    """
    device = images.device
    b = images.shape[0]
    unet = pipe.unet
    scheduler = pipe.scheduler

    mse_dict = {t: torch.zeros(b, device=device) for t in t_list}
    clip_dict = {t: torch.zeros(b, device=device) for t in t_list}

    # CLIP feature of original image (once)
    x_clip = to_clip_input(images, clip_mean, clip_std)
    with torch.autocast("cuda", enabled=(device.type=="cuda")):
        f_x = clip_model.encode_image(x_clip)
    f_x = f_x / (f_x.norm(dim=-1, keepdim=True) + 1e-12)

    for t in t_list:
        t_batch = torch.full((b,), t, device=device, dtype=torch.long)

        noise = torch.randn_like(images)
        x_t = scheduler.add_noise(images, noise, t_batch)
        eps_hat = unet(x_t, t_batch).sample

        # DDIM hardness
        mse = (noise - eps_hat).pow(2).mean(dim=(1,2,3))
        mse_dict[t] = mse

        # CLIP semantic proxy
        x0_hat = predict_x0_from_xt_eps(scheduler, x_t, eps_hat, t_batch)
        x0_clip = to_clip_input(x0_hat, clip_mean, clip_std)
        with torch.autocast("cuda", enabled=(device.type=="cuda")):
            f_r = clip_model.encode_image(x0_clip)
        f_r = f_r / (f_r.norm(dim=-1, keepdim=True) + 1e-12)
        clip_dict[t] = (1 - (f_x * f_r).sum(dim=-1)).float()

    return mse_dict, clip_dict


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","cifar100"], default="cifar100")
    ap.add_argument("--split", choices=["train","test"], default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_clip", default=None)
    ap.add_argument("--t_list", type=str, default="50,250,500,750")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", default="google/ddpm-cifar10-32")
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--clip_model", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="openai")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    t_list = [int(x) for x in args.t_list.split(",")]

    if args.out_clip is None:
        base, _ = os.path.splitext(args.out)
        args.out_clip = base + "_clip.npy"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*2-1),
    ])

    root = "./data"
    ds = CIFAR100(root, train=(args.split=="train"), download=True, transform=tfm) \
         if args.dataset=="cifar100" else \
         CIFAR10(root, train=(args.split=="train"), download=True, transform=tfm)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4,
        pin_memory=(device.type=="cuda")
    )

    print(f">>> Loading diffusion: {args.model_id}")
    pipe = DDPMPipeline.from_pretrained(args.model_id).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(args.ddim_steps)

    print(f">>> Loading CLIP: {args.clip_model}")
    clip_model, clip_mean, clip_std = load_openclip(
        device, args.clip_model, args.clip_pretrained
    )

    all_mse = {t: [] for t in t_list}
    all_clip = {t: [] for t in t_list}

    print(f">>> Fixed-t sweep: {t_list}")
    for x,_ in tqdm(loader, ncols=100):
        x = x.to(device, non_blocking=True)
        mse_d, clip_d = compute_ddim_hardness_and_clip(
            pipe, clip_model, clip_mean, clip_std, x, t_list
        )
        for t in t_list:
            all_mse[t].append(mse_d[t].cpu().numpy())
            all_clip[t].append(clip_d[t].cpu().numpy())

    # save
    for t in t_list:
        mse = np.concatenate(all_mse[t])
        clip = np.concatenate(all_clip[t])
        np.save(args.out.replace(".npy", f"_t{t}.npy"), mse)
        np.save(args.out_clip.replace(".npy", f"_t{t}.npy"), clip)
        print(f"[Saved] t={t}: mse/clip")

    print("✅ Done.")


if __name__ == "__main__":
    main()
