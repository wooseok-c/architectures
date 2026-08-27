"""
Seq2Seq (인코더-디코더), RNN으로 시퀀스를 다른 시퀀스로 (예: 번역).

구조:
  인코더 RNN: 입력 시퀀스 → 마지막 h = context (입력 요약 벡터 1개)
  디코더 RNN: context에서 시작 → 출력 단어를 한 스텝씩 생성
              각 스텝: 이전 상태 + 직전 출력단어 → 새 상태 → softmax로 단어 예측
              (직전 출력을 다음 입력으로 되먹임 = autoregressive)

학습까지는 안 하고, forward 구조 + shape 흐름만 확인.
        디코더 입력은 간단히 "직전 스텝 hidden"을 넘기는 식으로 단순화.
병목: 인코더 정보가 context 벡터 "하나"에 압축됨 → 나중에 attention이 개선.
"""
import torch
import torch.nn as nn
from rnn_layer import RNN
from rnn_cell import RNNCell

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.encoder = RNN(input_size, hidden_size)
        self.decoder_cell = RNNCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, max_len):
        """
        src: (batch, T_src, input_size)   입력 시퀀스
        max_len: 생성할 출력 길이
        반환: logits (batch, max_len, vocab_size)   각 스텝의 어휘 점수
        """
        batch = src.shape[0]
        _, context = self.encoder(src)          
        h = context
        dec_in = torch.zeros(batch, self.hidden_size)
        logits = []
        for t in range(max_len):
            h = self.decoder_cell(dec_in, h)
            score = self.out(h)
            logits.append(score)
            dec_in = h
        logits = torch.stack(logits, dim=1)
        return logits

