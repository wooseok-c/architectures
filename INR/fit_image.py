"""
SIREN 실험, 이미지 한 장을 좌표 함수로 표현하기

siren.py 의 TODO 를 다 채운 뒤 돌리세요.

  python fit_image.py --exp relu      # 1) Day 1: ReLU MLP 로 실패를 눈으로 보기
  python fit_image.py --exp compare   # 2) ReLU vs SIREN, 논문 Figure 1 재현 (f / ∇f / Δf)
  python fit_image.py --exp omega     # 3) omega_0 스윕 3 / 30 / 300
                                      #    학습은 저해상도, 렌더는 2배 → 픽셀 사이 거동 관찰

결과 png 는 이 폴더에 저장됩니다.

옵션: --side 128  --steps 500  --img /path/to/image.jpg
"""

import argparse
import io
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from siren import SIREN, ReLUMLP


#  데이터
def load_gray(side, path=None):
    """흑백 이미지 (side, side), 값 범위 [-1, 1]."""
    from PIL import Image
    if path is None:
        import matplotlib.cbook as cbook
        with cbook.get_sample_data("grace_hopper.jpg") as f:
            im = Image.open(io.BytesIO(f.read()))
    else:
        im = Image.open(path)
    im = im.convert("L")
    s = min(im.size)
    im = im.crop(((im.width - s) // 2, (im.height - s) // 2,
                  (im.width + s) // 2, (im.height + s) // 2))
    im = im.resize((side, side), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a * 2 - 1


def coord_grid(side, device):
    """[-1, 1]^2 격자 → (side*side, 2). 좌표 정규화는 SIREN 의 전제."""
    t = torch.linspace(-1, 1, side, device=device)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


#  학습 / 평가
def psnr(a, b):
    """a, b 는 [-1,1] 범위. 최대 진폭 2 기준."""
    mse = float(np.mean((a - b) ** 2))
    return 10 * math.log10(4.0 / max(mse, 1e-12))


def fit(model, coords, target, steps=500, lr=1e-4, log_every=100, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for i in range(steps + 1):
        out = model(coords)
        loss = ((out - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and i % log_every == 0:
            print(f"    {tag}step {i:5d}  loss {loss.item():.6f}")
    print(f"    {tag}{steps} steps in {time.time() - t0:.1f}s")
    return model


@torch.no_grad()
def render(model, side, device):
    out = model(coord_grid(side, device))
    return out.reshape(side, side).detach().cpu().numpy()


def derivatives(model, side, device):
    """∇f 의 크기와 라플라시안 Δf. 이중 역전파라 느립니다."""
    coords = coord_grid(side, device).requires_grad_(True)
    y = model(coords)
    g = torch.autograd.grad(y.sum(), coords, create_graph=True)[0]      # (N, 2)
    lap = 0.0
    for d in range(2):
        gd = torch.autograd.grad(g[:, d].sum(), coords, create_graph=True)[0]
        lap = lap + gd[:, d]
    gmag = g.norm(dim=-1).detach().cpu().numpy().reshape(side, side)
    lap = lap.detach().cpu().numpy().reshape(side, side)
    return gmag, lap


def show(ax, a, title, cmap="gray", sym=False):
    if sym:
        v = np.percentile(np.abs(a), 99) + 1e-8
        ax.imshow(a, cmap="RdBu", vmin=-v, vmax=v)
    else:
        v0, v1 = np.percentile(a, 1), np.percentile(a, 99)
        ax.imshow(a, cmap=cmap, vmin=v0, vmax=v1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


#  실험
def exp_relu(args, dev, img, coords, target):
    print("\n[1) ReLU MLP 로 이미지 맞추기]  뭐가 먼저 맞고 뭐가 안 맞는지 보세요")
    torch.manual_seed(0)
    m = ReLUMLP(2, 256, 3, 1).to(dev)
    fit(m, coords, target, args.steps, args.lr, tag="relu ")
    rec = render(m, args.side, dev)
    print(f"    PSNR {psnr(img, rec):.2f} dB")

    fig, ax = plt.subplots(1, 2, figsize=(6, 3.2))
    show(ax[0], img, "ground truth")
    show(ax[1], rec, f"ReLU MLP  {psnr(img, rec):.1f} dB")
    fig.tight_layout(); fig.savefig("out_relu.png", dpi=140)
    print("    → out_relu.png")


def exp_compare(args, dev, img, coords, target):
    print("\n[2) Figure 1 재현]  이미지 값만 지도하고, 도함수는 그냥 계산해서 봅니다")
    models = {}
    torch.manual_seed(0)
    models["ReLU MLP"] = ReLUMLP(2, 256, 3, 1).to(dev)
    torch.manual_seed(0)
    models[f"SIREN (w0={args.omega0:g})"] = SIREN(
        2, 256, 3, 1, first_omega_0=args.omega0, hidden_omega_0=args.omega0).to(dev)

    fig, axes = plt.subplots(3, 3, figsize=(8.4, 8.4))
    gt_g, gt_l = np.gradient(img)
    show(axes[0][0], img, "ground truth")
    show(axes[1][0], np.hypot(gt_g, gt_l), "|grad f|  (finite diff)")
    show(axes[2][0], np.gradient(np.gradient(img)[0])[0] + np.gradient(np.gradient(img)[1])[1],
         "laplacian f  (finite diff)", sym=True)

    for col, (name, m) in enumerate(models.items(), start=1):
        fit(m, coords, target, args.steps, args.lr, tag=f"{name} ")
        rec = render(m, args.side, dev)
        gmag, lap = derivatives(m, args.side, dev)
        show(axes[0][col], rec, f"{name}   {psnr(img, rec):.1f} dB")
        show(axes[1][col], gmag, "|grad f|")
        show(axes[2][col], lap, "laplacian f", sym=True)
        print(f"    {name}: PSNR {psnr(img, rec):.2f} dB, |Δf| 평균 {np.abs(lap).mean():.3e}")

    fig.tight_layout(); fig.savefig("out_figure1.png", dpi=140)
    print("    → out_figure1.png   (2·3행에서 갈리는지 보세요)")


def exp_omega(args, dev, img, coords, target):
    print("\n[3) omega_0 스윕]  loss 와 화질이 어긋나는 행이 있는지")
    up = args.side * 2
    fig, axes = plt.subplots(2, len(args.omegas), figsize=(3.0 * len(args.omegas), 6.2))
    if len(args.omegas) == 1:
        axes = axes.reshape(2, 1)

    print(f"\n    {'w0':>6} | {'train loss':>11} | {'PSNR(학습해상도)':>16}")
    for j, w0 in enumerate(args.omegas):
        torch.manual_seed(0)
        m = SIREN(2, 256, 3, 1, first_omega_0=w0, hidden_omega_0=w0).to(dev)
        fit(m, coords, target, args.steps, args.lr, log_every=0, tag=f"w0={w0:g} ")
        rec = render(m, args.side, dev)
        with torch.no_grad():
            tl = float(((m(coords) - target) ** 2).mean())
        big = render(m, up, dev)
        print(f"    {w0:6g} | {tl:11.3e} | {psnr(img, rec):16.2f}")
        show(axes[0][j], rec, f"w0={w0:g}   trained @ {args.side}   {psnr(img, rec):.1f} dB")
        show(axes[1][j], big, f"rendered @ {up}  (unseen coords)")

    fig.tight_layout(); fig.savefig("out_omega.png", dpi=140)
    print("\n    → out_omega.png")
    print("    아랫줄이 핵심입니다. loss 가 낮은데 아랫줄이 지저분하면,")
    print("    'loss 는 격자 위에서만 재는데 ω0 는 그 사이를 정한다' 를 눈으로 본 겁니다.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="compare", choices=["relu", "compare", "omega"])
    p.add_argument("--side", type=int, default=128)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--omega0", type=float, default=30.0)
    p.add_argument("--omegas", type=float, nargs="+", default=[3, 30, 300])
    p.add_argument("--img", default=None)
    args = p.parse_args()

    dev = pick_device()
    print(f"device: {dev}   side: {args.side}   steps: {args.steps}")

    img = load_gray(args.side, args.img)
    coords = coord_grid(args.side, dev)
    target = torch.from_numpy(img.reshape(-1, 1)).to(dev)

    {"relu": exp_relu, "compare": exp_compare, "omega": exp_omega}[args.exp](
        args, dev, img, coords, target)


if __name__ == "__main__":
    main()
