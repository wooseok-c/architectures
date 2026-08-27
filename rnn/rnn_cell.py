"""
RNN 셀, 시퀀스 한 스텝을 처리하는 기본 단위.

핵심 식:
    h_t = tanh( W_x · x_t  +  W_h · h_{t-1}  +  b )
          └── 현재 입력 ──┘   └── 이전 기억 ──┘

- x_t     : 현재 스텝 입력 (예: 단어 벡터), shape (batch, input_size)
- h_{t-1} : 이전 hidden state (기억),       shape (batch, hidden_size)
- h_t     : 새 hidden state,                shape (batch, hidden_size)
- W_x, W_h, b : 학습되는 가중치. "모든 스텝에서 같은 것 재사용" (파라미터 공유)
"""
import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # nn.Linear가 affine 연산
        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias = False)
    # h_prev : 이전 스텝의 hidden state, h_{t-1}
    def forward(self, x_t, h_prev):
        """한 스텝: (x_t, h_{t-1}) -> h_t"""
        h_t = torch.tanh(self.W_x(x_t) + self.W_h(h_prev))
        return h_t
