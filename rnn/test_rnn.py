"""
RNN 단계별 검증 — shape이 맞게 흐르는지.
실행:  python test_rnn.py
"""
import torch
from rnn_cell import RNNCell
from rnn_layer import RNN
from seq2seq import Seq2Seq

batch, T, input_size, hidden_size, vocab = 2, 5, 10, 16, 100

# ① RNN 셀: (x_t, h_prev) -> h_t
cell = RNNCell(input_size, hidden_size)
x_t = torch.zeros(batch, input_size)
h_prev = torch.zeros(batch, hidden_size)
h_t = cell(x_t, h_prev)
print("① RNNCell  h_t :", tuple(h_t.shape), " 기대 (2, 16)")
assert h_t.shape == (batch, hidden_size)

# ② RNN 레이어: 시퀀스 -> 모든 h + 마지막 h
rnn = RNN(input_size, hidden_size)
x = torch.zeros(batch, T, input_size)
outputs, h = rnn(x)
print("② RNN outputs :", tuple(outputs.shape), " 기대 (2, 5, 16)")
print("   RNN last h  :", tuple(h.shape), "     기대 (2, 16)")
assert outputs.shape == (batch, T, hidden_size)
assert h.shape == (batch, hidden_size)

# ③ Seq2Seq: 입력 시퀀스 -> 출력 logits
model = Seq2Seq(input_size, hidden_size, vocab)
src = torch.zeros(batch, T, input_size)
logits = model(src, max_len=4)
print("③ Seq2Seq log :", tuple(logits.shape), " 기대 (2, 4, 100)")
assert logits.shape == (batch, 4, vocab)

print("\nOK — RNN 셀 / RNN 레이어 / Seq2Seq 모두 통과!")
