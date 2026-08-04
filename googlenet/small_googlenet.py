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
        ############################################################
        # TODO: 아래 구조로 레이어를 정의하세요 (채널 수는 자유롭게 조절 가능)
        #   self.stem = conv_block(3, 64, kernel_size=3, padding=1)   # 입력 처리
        #   self.inc1 = Inception(64,  32, 32, 64, 8, 16, 16)         # 출력채널 = 32+64+16+16 = 128
        #   self.inc2 = Inception(128, 64, 64, 96, 16, 32, 32)        # 출력채널 = 64+96+32+32 = 224
        #   self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling: (N,C,H,W)->(N,C,1,1)
        #   self.fc   = nn.Linear(224, num_classes)
        # 주의: 각 Inception의 in_ch = 앞 층의 출력채널 합 (차원 맞물림!)
        ############################################################

        ############################################################
        #                    END OF YOUR CODE                      #
        ############################################################

    def forward(self, x):
        ############################################################
        # TODO: stem -> inc1 -> inc2 -> global avg pool -> flatten -> fc
        #   x = self.stem(x)
        #   x = self.inc1(x); x = self.inc2(x)
        #   x = self.pool(x)              # (N, C, 1, 1)
        #   x = torch.flatten(x, 1)        # (N, C)
        #   scores = self.fc(x)
        ############################################################
        pass


if __name__ == "__main__":
    m = SmallGoogLeNet(num_classes=10)
    x = torch.zeros(2, 3, 32, 32)
    print(m(x).shape)   # 기대: torch.Size([2, 10])
