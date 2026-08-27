"""
BasicBlock 검증, 3가지 경우가 다 맞게 흐르는지.
실행:  python test_basic_block.py

경우 1: 차원 같음 (in=out=64, stride=1) -> identity shortcut, 크기 유지
경우 2: 채널만 바뀜 (64->128, stride=1)  -> 1x1 conv shortcut, 크기 유지
경우 3: 채널+크기 바뀜 (64->128, stride=2) -> 1x1 conv(stride2) shortcut, 크기 절반
"""
import torch
from basic_block import BasicBlock

x = torch.zeros(2, 64, 32, 32)   # (N, C, H, W)

# 1) identity shortcut
b1 = BasicBlock(64, 64, stride=1)
o1 = b1(x)
print("1) 64->64, s1 :", o1.shape, " 기대 (2, 64, 32, 32)")
assert o1.shape == (2, 64, 32, 32)

# 2) projection shortcut (채널만)
b2 = BasicBlock(64, 128, stride=1)
o2 = b2(x)
print("2) 64->128,s1 :", o2.shape, " 기대 (2, 128, 32, 32)")
assert o2.shape == (2, 128, 32, 32)

# 3) projection + downsample (채널+크기)
b3 = BasicBlock(64, 128, stride=2)
o3 = b3(x)
print("3) 64->128,s2 :", o3.shape, " 기대 (2, 128, 16, 16)")
assert o3.shape == (2, 128, 16, 16)

print("\nOK, BasicBlock 3가지 경우 모두 통과!")
