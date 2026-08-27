"""
Bahdanau (additive) attention 구현 연습 (PyTorch nn.Module).

어텐션 = 디코더 상태 s 와 인코더 H(annotations) 로 문맥 c 를 만든다.
    1) 점수:   e_j = v_a^T tanh(W_a s + U_a h_j)     (원문 단어 j 마다 점수 하나)
    2) 주목도: α   = softmax(e)                       (원문 위치들에 대해, 합=1)
    3) 문맥:   c   = Σ_j α_j h_j                      (annotation 가중합)

어텐션에서 weight 가진 건 정렬모델 a 하나뿐:
    W_a (align_dim, dec_dim),  U_a (align_dim, enc_dim),  v_a (align_dim,)
softmax 와 가중합에는 파라미터 없음.
"""
import math
import torch
import torch.nn as nn

class MyBahdanauAttention(nn.Module):
    def __init__(self, dec_dim, enc_dim, align_dim):
        """
        dec_dim   = n   디코더 상태 s 의 차원
        enc_dim   = 2n  annotation h_j 의 차원 (양방향 bi-RNN concat)
        align_dim = n'  정렬모델 내부 차원
        """
        super().__init__()
        # 둘 다 행(내는 쪽) = align_dim 이라 더할 수 있음. 열(받는 쪽)만 다름.
        self.W_a = nn.Parameter(torch.empty(align_dim, dec_dim))   # (n', n)  : s 변환
        self.U_a = nn.Parameter(torch.empty(align_dim, enc_dim))   # (n', 2n) : h 변환
        self.v_a = nn.Parameter(torch.empty(align_dim))            # (n',)    : 점수 스칼라로
        std = 1.0 / math.sqrt(align_dim)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, s, H):
        """
        s: (N, dec_dim)        디코더 상태 s_{i-1}
        H: (N, Tx, enc_dim)    인코더 annotations (원문 단어 Tx 개)
        return: c (N, enc_dim),  alpha (N, Tx)
        """
        Wa_s = s @ self.W_a.t()          # (N, align_dim)       = W_a s
        Ua_H = H @ self.U_a.t()          # (N, Tx, align_dim)   = U_a h_j  (모든 j 한 방에; U_a h_j 미리 계산!)
        # W_a s 를 (N, 1, align_dim) 으로 늘려 모든 j 에 broadcasting 하여 더함
        pre = torch.tanh(Wa_s.unsqueeze(1) + Ua_H)   # (N, Tx, align_dim)
        e = pre @ self.v_a
        alpha = torch.softmax(e, dim = 1)
        c = (alpha.unsqueeze(-1) * H).sum(dim = 1)

        return c, alpha

# 검증
def _reference(att, s, H):
    """loop"""
    N, Tx, E = H.shape
    e = torch.zeros(N, Tx, dtype=H.dtype)
    for n in range(N):
        for j in range(Tx):
            pre = torch.tanh(att.W_a @ s[n] + att.U_a @ H[n, j])  # (align_dim,)
            e[n, j] = att.v_a @ pre                                # 스칼라
    alpha = torch.softmax(e, dim=1)
    c = torch.zeros(N, E, dtype=H.dtype)
    for n in range(N):
        for j in range(Tx):
            c[n] += alpha[n, j] * H[n, j]
    return c, alpha

def check_attention():
    torch.manual_seed(0)
    N, Tx, dec_dim, enc_dim, align_dim = 3, 5, 4, 6, 7
    s = torch.randn(N, dec_dim, dtype=torch.double)
    H = torch.randn(N, Tx, enc_dim, dtype=torch.double)

    att = MyBahdanauAttention(dec_dim, enc_dim, align_dim).double()
    c, alpha = att(s, H)
    c_ref, alpha_ref = _reference(att, s, H)

    shape_ok = (c.shape == (N, enc_dim)) and (alpha.shape == (N, Tx))
    sum_ok = torch.allclose(alpha.sum(dim=1), torch.ones(N, dtype=torch.double), atol=1e-8)
    match_ok = torch.allclose(c, c_ref, atol=1e-8) and torch.allclose(alpha, alpha_ref, atol=1e-8)

    ok = shape_ok and sum_ok and match_ok
    print(f"Attention {'PASS' if ok else 'FAIL'}"
          f"  (shape={'ok' if shape_ok else 'X'}, α합=1 {'ok' if sum_ok else 'X'}, 정답일치 {'ok' if match_ok else 'X'})")
    if not match_ok and c.shape == (N, enc_dim):
        print("  c max diff:", (c - c_ref).abs().max().item())


if __name__ == "__main__":
    check_attention()
