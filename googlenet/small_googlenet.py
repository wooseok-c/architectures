"""
(선택 도전) 미니 GoogLeNet — Inception module을 몇 개 쌓아 CIFAR-10 분류.
전체 22층은 과하니, 핵심 흐름만: stem -> inception 몇 개 -> global avg pool -> FC.

먼저 inception.py를 완성한 뒤에 시도하세요.
"""
import torch
import torch.nn as nn
from inception import Inception, conv_block


class SmallGoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 주의: 각 Inception의 in_ch = 앞 층의 출력채널 합 (차원 맞물림!)


    def forward(self, x):
        pass


if __name__ == "__main__":
    m = SmallGoogLeNet(num_classes=10)
    x = torch.zeros(2, 3, 32, 32)
    print(m(x).shape)   # 기대: torch.Size([2, 10])
