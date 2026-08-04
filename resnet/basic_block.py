"""
ResNet BasicBlock — ResNet의 핵심 부품.
논문: "Deep Residual Learning for Image Recognition" (He et al., 2015), Section 3.1~3.2, Figure 2.

핵심:  y = F(x) + x       (출력 = 잔차 + 입력)
  F(x) = conv -> BN -> relu -> conv -> BN   (잔차 함수)
  + x  = shortcut (입력을 그대로 더함, 파라미터 0)
  마지막에 relu 한 번 더

차원이 다를 때 (채널 늘거나 stride로 크기 줄일 때):
  y = F(x) + Ws(x)  (Ws = 1x1 conv, 필요시 stride로 x를 F(x)와 같은 shape으로)

주의: sum(+)이라 F(x)와 shortcut의 채널·H·W가 "완전히 같아야" 함.
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        """
        - in_ch : 입력 채널
        - out_ch: 출력 채널
        - stride: 1이면 크기 유지, 2면 크기 절반 (다운샘플링)
        차원이 바뀌면 (in_ch != out_ch 또는 stride != 1) shortcut에 1x1 conv 필요.
        """
        super().__init__()
        ############################################################
        # TODO: F(x) 경로 정의 — conv3x3 -> BN -> relu -> conv3x3 -> BN
        #   self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        #   self.bn1   = nn.BatchNorm2d(out_ch)
        #   self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        #   self.bn2   = nn.BatchNorm2d(out_ch)
        #   self.relu  = nn.ReLU()
        # (첫 conv에만 stride 적용 → 크기는 여기서 줄임. bias=False는 BN이 있어서 관례)
        ############################################################
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_ch)
            ) # 여기까지 F(x)
        ############################################################
        # TODO: shortcut 정의 — 차원이 같으면 그냥 통과(identity), 다르면 1x1 conv
        #   if stride != 1 or in_ch != out_ch:
        #       self.shortcut = nn.Sequential(
        #           nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
        #           nn.BatchNorm2d(out_ch),
        #       )
        #   else:
        #       self.shortcut = nn.Identity()   # 그대로 통과 (F(x)+x)
        ############################################################
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride = stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        
        self.relu = nn.ReLU()
        ############################################################
        #                    END OF YOUR CODE                      #
        ############################################################

    def forward(self, x):
        ############################################################
        # TODO:
        #   out = self.conv1(x); out = self.bn1(out); out = self.relu(out)   # conv-bn-relu
        #   out = self.conv2(out); out = self.bn2(out)                       # conv-bn (relu 아직 X)
        #   out = out + self.shortcut(x)   # F(x) + shortcut(x)  ← 핵심!
        #   out = self.relu(out)           # 더한 뒤 relu
        #   return out
        # 왜 shortcut(x)? 차원 같으면 x 그대로, 다르면 1x1 conv 거친 x
        ############################################################
        out = self.residual(x)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out
