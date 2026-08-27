"""
Inception module, GoogLeNet의 핵심 부품.
논문: "Going Deeper with Convolutions" (Szegedy et al., 2014), Section 4, Figure 2(b).

Inception module = 4개의 병렬 branch를 "채널 방향"으로 concat:

  입력 x
    ├─ branch1: 1x1 conv
    ├─ branch2: 1x1 conv(축소) -> 3x3 conv
    ├─ branch3: 1x1 conv(축소) -> 5x5 conv
    └─ branch4: 3x3 maxpool    -> 1x1 conv
    => torch.cat([b1, b2, b3, b4], dim=1)   # 채널(C) 방향으로 이어붙임

핵심 아이디어:
  - 여러 필터 크기(1x1,3x3,5x5)를 "동시에" 써서 다양한 크기의 특징을 뽑음
  - 1x1 conv = 채널 수를 줄이는 bottleneck (5x5 앞에 두면 계산량 대폭 절감)
  - 모든 branch는 H,W를 유지(padding) → concat 가능 (채널만 늘어남)
"""
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, kernel_size, padding=0):
    """conv -> ReLU 묶음. (원한다면 BN을 여기 nn.Conv2d 다음에 추가해볼 것)"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
    )


class Inception(nn.Module):
    def __init__(self, in_ch, c1, c3_reduce, c3, c5_reduce, c5, pool_proj):
        """
        채널 수 인자 (논문 Table 1의 값들):
        - in_ch    : 입력 채널 수
        - c1       : branch1의 1x1 출력 채널
        - c3_reduce: branch2의 1x1 축소 채널,  c3: 3x3 출력 채널
        - c5_reduce: branch3의 1x1 축소 채널,  c5: 5x5 출력 채널
        - pool_proj: branch4의 1x1 출력 채널
        출력 채널 = c1 + c3 + c5 + pool_proj
        """
        super().__init__()
        #   여러 층 묶을 땐 nn.Sequential(conv_block(...), conv_block(...))
        self.branch1 = conv_block(in_ch, c1, kernel_size=1)
        self.branch2 = nn.Sequential(conv_block(in_ch, c3_reduce, kernel_size=1),
                                     conv_block(c3_reduce, c3, kernel_size=3, padding = 1)
                                     )
        self.branch3 = nn.Sequential(conv_block(in_ch, c5_reduce, kernel_size=1),
                                      conv_block(c5_reduce, c5, kernel_size=5, padding = 2)
                                    )
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
                                     conv_block(in_ch, pool_proj, kernel_size=1))


    def forward(self, x):
        # 왜 dim=1? 텐서가 (N, C, H, W)라 C(채널)가 1번 축.
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim = 1)
        pass
