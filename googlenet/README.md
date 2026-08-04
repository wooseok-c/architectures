# GoogLeNet (Inception) 구현 연습

논문: "Going Deeper with Convolutions" (Szegedy et al., 2014)

## 목표
GoogLeNet의 핵심 부품인 **Inception module**을 직접 구현한다.
전체 22층 GoogLeNet을 다 짤 필요는 없다 — Inception module 하나를 이해/구현하면
GoogLeNet의 핵심은 다 잡은 것. (전체망은 이걸 9번 쌓은 것)

## 핵심 개념 2가지
1. **Inception module** = 여러 필터(1x1, 3x3, 5x5, pool)를 병렬로 적용 후 채널 concat
   → 다양한 크기의 특징을 한 층에서 동시에 추출
2. **1x1 convolution** = 채널 수를 줄이는 bottleneck (계산량 절감)
   → ResNet bottleneck, Transformer FFN으로 이어지는 중요 아이디어

## 파일
- `inception.py`    — Inception module 스켈레톤 (여기를 채운다) ★
- `test_inception.py` — shape 검증 (python test_inception.py)
- `small_googlenet.py` — (선택) inception 몇 개를 쌓은 미니 GoogLeNet 스켈레톤

## 진행 순서
1. `inception.py`의 `__init__`에서 4개 branch 정의
2. `forward`에서 4개 branch 통과 후 `torch.cat(..., dim=1)`
3. `python test_inception.py` 실행 → `torch.Size([2, 256, 28, 28])` 나오면 성공
4. (선택) `small_googlenet.py`로 여러 inception을 쌓아 CIFAR 분류

## 실행 환경
cs231n venv 재사용 (torch 설치돼 있음):
  python test_inception.py

## 논문에서 읽을 곳
- Section 4 Architectural Details + Figure 2 (Inception module) ★★
- Section 5 + Table 1 (전체 구조, 훑기)
