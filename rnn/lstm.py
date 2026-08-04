"""
LSTM 셀 한 스텝 구현 

검증: torch.nn.LSTMCell 에 같은 weight를 복사해서 출력이 일치하는지 확인.
실행: python lstm.py   ->  "LSTM forward PASS" 나오면 성공.

수식 (PyTorch 관례, 게이트 순서 [i, f, g, o]):
    i = σ(pre_i)                # input gate
    f = σ(pre_f)                # forget gate
    g = tanh(pre_g)             # candidate cell
    o = σ(pre_o)                # output gate
    c_new = f ⊙ c + i ⊙ g       # cell 업데이트 (f⊙c 가 기억 하이웨이)
    h_new = o ⊙ tanh(c_new)     # hidden 출력 (새 cell 을 씀!)
"""
import math
import torch.nn as nn
import torch

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # torch 레이아웃: 게이트 [i, f, g, o] 를 세로로 쌓은 큰 행렬.
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        std = 1.0 / math.sqrt(hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x, state):
        """
        x: (N, I)
        state: (h, c),  h,c: (N, H)
        return: (h_new, c_new)
        """
        H = self.hidden_size
        h, c = state

        # 게이트 4개짜리 pre-activation. gates: (N, 4H)
        gates = x @ self.weight_ih.t() + self.bias_ih \
              + h @ self.weight_hh.t() + self.bias_hh
        # [i, f, g, o] 로 잘라둠. 각각 (N, H), affine layer 계산 후 각각 activation f이 달라서 슬라이싱
        pre_i = gates[:, 0:H]
        pre_f = gates[:, H:2 * H]
        pre_g = gates[:, 2 * H:3 * H]
        pre_o = gates[:, 3 * H:4 * H]
        ###############################################################
        # TODO: LSTM 한 스텝을 구현하세요.                                 #
        ###############################################################
        i = torch.sigmoid(pre_i)
        f = torch.sigmoid(pre_f)
        g = torch.tanh(pre_g)
        o = torch.sigmoid(pre_o)
        c_new = f * c + i * g # 옛날 것(조절) + 새 것(조절)
        h_new = o * torch.tanh(c_new)
        # *****END OF YOUR CODE*****
        ###############################################################
        return h_new, c_new

# 검증 (건드릴 필요 없음)
def check_lstm():
    torch.manual_seed(0)
    N, I, H = 4, 5, 6
    x = torch.randn(N, I, dtype=torch.double)
    h = torch.randn(N, H, dtype=torch.double)
    c = torch.randn(N, H, dtype=torch.double)

    ref = nn.LSTMCell(I, H).double()
    mine = MyLSTMCell(I, H).double()
    with torch.no_grad():
        mine.weight_ih.copy_(ref.weight_ih)
        mine.weight_hh.copy_(ref.weight_hh)
        mine.bias_ih.copy_(ref.bias_ih)
        mine.bias_hh.copy_(ref.bias_hh)

    h_ref, c_ref = ref(x, (h, c))
    h_mine, c_mine = mine(x, (h, c))
    ok = torch.allclose(h_ref, h_mine, atol=1e-8) and torch.allclose(c_ref, c_mine, atol=1e-8)
    print(f"LSTM forward {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  h max diff:", (h_ref - h_mine).abs().max().item())
        print("  c max diff:", (c_ref - c_mine).abs().max().item())


if __name__ == "__main__":
    check_lstm()
