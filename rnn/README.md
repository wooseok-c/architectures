# RNN / Seq2Seq 구현 연습 (방식 B: 구조·shape 위주)

## 목표
RNN이 시퀀스를 어떻게 처리하는지 손으로 느끼기.
셀 → 시퀀스 → 인코더-디코더 순으로 쌓아 올린다. (학습까지는 X, forward 구조만)

## 핵심 식
    RNN 셀:  h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)
    같은 W를 모든 스텝에 재사용 (파라미터 공유)

## 파일 (순서대로 채우기)
1. `rnn_cell.py`   — RNN 셀 한 스텝 (h_t = tanh(...))  ★ 여기부터
2. `rnn_layer.py`  — 시퀀스 전체 (for 루프로 셀 반복)
3. `seq2seq.py`    — 인코더-디코더 (context → 디코더 생성)
4. `test_rnn.py`   — 세 단계 shape 검증

## 진행
1~3을 채우고:
    python test_rnn.py
  → 세 단계 다 통과하면 성공

## 개념 연결
- RNN 셀 = affine 2개(현재 입력 + 이전 기억) + tanh
- 시퀀스 = 셀을 시간축으로 반복 (h를 계속 넘김 = 기억 사슬)
- 인코더 마지막 h = context (입력 요약 벡터 1개) ← 병목
- 디코더 = context에서 한 단어씩 생성 (직전 출력 되먹임 = autoregressive)
- 다음(내일): 이 병목을 Attention이 개선 (context 하나 → 모든 h 직접 보기)
