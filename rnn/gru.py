"""
GRU 셀 한 스텝 구현 연습 (PyTorch nn.Module).

아래 forward 안의 #### TODO 박스 채우기
검증: torch.nn.GRUCell 에 같은 weight를 복사해서 출력이 일치하는지 확인. (weight sharing)
실행: python gru.py   ->  "GRU forward PASS" 나오면 성공

수식
    r = σ(i_r + h_r)                 # reset gate
    z = σ(i_z + h_z)                 # update gate
    n = tanh(i_n + r ⊙ h_n)          # candidate  (r은 h_n 에만 곱함!)
    h_new = (1 - z) ⊙ n + z ⊙ h      # blend  (z가 '과거 유지' 비율)
"""
import math
import torch
import torch.nn as nn

class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # torch 레이아웃: 게이트 [r, z, n] 을 세로로 쌓은 큰 행렬.
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        std = 1.0 / math.sqrt(hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x, h):
        """
        x: (N, I)  현재 입력, N: 문장 개수
        h: (N, H)  이전 hidden
        return h_new: (N, H)
        """
        H = self.hidden_size

        # 입력쪽 / 은닉쪽 pre-activation
        gi = x @ self.weight_ih.t() + self.bias_ih   # (N, 3H)
        gh = h @ self.weight_hh.t() + self.bias_hh   # (N, 3H)

        # [r, z, n] 세 덩어리로 잘라둠. 각각 (N, H)
        i_r, i_z, i_n = gi[:, 0:H], gi[:, H:2 * H], gi[:, 2 * H:3 * H]
        h_r, h_z, h_n = gh[:, 0:H], gh[:, H:2 * H], gh[:, 2 * H:3 * H]

        ###############################################################
        # TODO: GRU 한 스텝을 구현 (torch.sigmoid, torch.tanh 사용) #                           #
        ###############################################################   
        r = torch.sigmoid(i_r + h_r) # 재료 준비, 후보 n을 만들 때 과거 섞기
        z = torch.sigmoid(i_z + h_z) # 최종 결정, 과거 유지 vs 후보 채택
        n = torch.tanh(i_n + r * h_n) # r을 만들어야 n을 구현, r을 반영한 새로운 초안
        h_new = (1-z) * n + z * h
        # *END OF YOUR CODE*
        ###############################################################
        return h_new

# 검증 
def check_gru():
    torch.manual_seed(0)
    N, I, H = 4, 5, 6
    x = torch.randn(N, I, dtype=torch.double)
    h = torch.randn(N, H, dtype=torch.double)

    ref = nn.GRUCell(I, H).double()
    mine = MyGRUCell(I, H).double()
    with torch.no_grad():
        mine.weight_ih.copy_(ref.weight_ih)
        mine.weight_hh.copy_(ref.weight_hh)
        mine.bias_ih.copy_(ref.bias_ih)
        mine.bias_hh.copy_(ref.bias_hh)

    out_ref = ref(x, h)
    out_mine = mine(x, h)
    ok = torch.allclose(out_ref, out_mine, atol=1e-8)
    print(f"GRU forward {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  max diff:", (out_ref - out_mine).abs().max().item())


if __name__ == "__main__":
    check_gru()
