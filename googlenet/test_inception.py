"""
Inception module 검증 — shape이 맞게 흐르는지 확인.
실행:  python test_inception.py
성공:  torch.Size([2, 256, 28, 28]) 출력

논문 inception(3a)의 채널 구성을 그대로 씀:
  in=192, 1x1=64, 3x3reduce=96, 3x3=128, 5x5reduce=16, 5x5=32, pool_proj=32
  출력 채널 = 64 + 128 + 32 + 32 = 256  (H,W는 유지)
"""
import torch
from inception import Inception

m = Inception(in_ch=192, c1=64, c3_reduce=96, c3=128,
              c5_reduce=16, c5=32, pool_proj=32)
x = torch.zeros(2, 192, 28, 28)   # (N, C, H, W)
out = m(x)
print(out.shape)                  # 기대: torch.Size([2, 256, 28, 28])

assert out.shape == (2, 256, 28, 28), \
    f"채널/크기 안 맞음: {out.shape} (기대 (2,256,28,28))"
print("OK — Inception module 통과!")
