# ResNet BasicBlock 구현 연습

논문: "Deep Residual Learning for Image Recognition" (He et al., 2015)

## 핵심 아이디어
- **residual (잔차) 학습**: 층이 H(x) 전체 대신 잔차 F(x)=H(x)-x 만 배움
- **결과 식**:  y = F(x) + x   (입력 x를 shortcut으로 더함)
- **왜 좋은가**: identity가 필요하면 F=0(가중치 0)만 만들면 됨 → 깊은 망 학습 가능
- **+x** = 트랜스포머 "Add & Norm"의 Add로 이어지는 부품 (ViT/Swin/SwiFT의 기반)

## sum vs concat (GoogLeNet과 대조)
- Inception: concat(채널 쌓기) → H·W만 같으면 됨
- ResNet: sum(더하기) → 채널·H·W 다 같아야 함
  → 차원 다르면 shortcut에 1x1 conv(Ws)로 x를 맞춤 (+필요시 stride)

## 파일
- `basic_block.py`       — BasicBlock 스켈레톤 (여기 채움) ★
- `test_basic_block.py`  — 3경우 shape 검증

## 진행 순서
1. `basic_block.py`의 __init__: F(x) 경로(conv-BN-relu-conv-BN) + shortcut(identity 또는 1x1 conv)
2. forward: out = F(x) + shortcut(x) 후 relu
3. 검증:
   python test_basic_block.py
   → 3경우 모두 통과하면 성공

## 3가지 shortcut 경우
① 차원 같음 (in=out, stride=1) → identity: y = F(x) + x
② 채널만 다름 → 1x1 conv(stride1): y = F(x) + Ws·x
③ 채널+크기 다름 → 1x1 conv(stride2): 크기도 절반으로 맞춤

## 논문에서 읽을 곳
- Section 1 (degradation 문제), Section 3.1~3.2 (residual) + Figure 2 ★
