"""
RNN 레이어 — 시퀀스 전체를 처리 (RNN 셀을 시간축으로 반복).

시퀀스 x = [x_1, x_2, ..., x_T] 를 받아,
h_0 = 0 에서 시작해 각 스텝마다 셀을 돌려 h를 갱신한다:
    h_1 = cell(x_1, h_0)
    h_2 = cell(x_2, h_1)
    ...
    h_T = cell(x_T, h_{T-1})

- 같은 cell(같은 W)을 매 스텝 재사용 = 파라미터 공유
- 출력: 모든 스텝의 h (outputs) + 마지막 h (= 시퀀스 요약, "context"의 원형)
"""
import torch
import torch.nn as nn
from rnn_cell import RNNCell


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size)   # 셀 하나를 재사용

    def forward(self, x):
        """
        x: (batch, T, input_size)   시퀀스 (T = 길이)
        반환:
          outputs: (batch, T, hidden_size)  매 스텝의 h
          h:       (batch, hidden_size)      마지막 h (시퀀스 요약)
        """
        batch, T, _ = x.shape
        h = torch.zeros(batch, self.hidden_size)   # h_0 = 0 (빈 기억)
        outputs = []
        # 왜 h를 계속 덮어쓰나? 이전 h를 다음 스텝에 넘겨야 하니까 (기억 사슬)
        for t in range(T):
            x_t = x[:, t, :] # 축0, 축1, 축2 -> 문장1, 문장2 
            h = self.cell(x_t, h)
            outputs.append(h)
        outputs = torch.stack(outputs, dim =1)
        return outputs, h