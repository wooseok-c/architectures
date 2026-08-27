"""
Transformer attention 구현

두 개 채우기:
  1) scaled_dot_product_attention    식 + 스케일/마스크
  2) MyMultiHeadAttention            헤드로 쪼개 각자 어텐션 → Concat.

검증: python attention.py
  - 1) torch.nn.functional.scaled_dot_product_attention 과 비교 (마스크 없음/causal)
  - 2) torch.nn.MultiheadAttention 에 같은 weight 복사해서 비교

수식:
  1) Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + mask) V
  2) MultiHead = Concat(head_1..head_h) W^O,  head_i = Attention(Q_i, K_i, V_i)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (..., L, d_k)   (앞쪽 차원은 batch나 head 등 뭐든; 마지막 두 개가 L, d_k)
    mask:    (..., L, L) 또는 None.  가릴 위치에 -inf 가 든 '덧셈' 마스크.
    return:  out (..., L, d_k),  alpha (..., L, L)
    """
    d_k = Q.size(-1) # 마지막 차원

    # 이전에 attention 에서 바뀐 부분을 활용                             #

    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k) # 맨 뒤에 L과 d_k 만 뒤집기
    if mask is not None:
        scores = scores + mask # mask가 텐서여서 이렇게 작성
    alpha = F.softmax(scores, -1)
    out = alpha @ V

    return out, alpha

# 2) Multi-Head Attention
class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 은 num_heads 로 나눠떨어져야 함"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads   # 각 헤드 차원 (d_model/h)
        # 입력 투영 3개 + 출력 투영 1개 (torch.nn.MultiheadAttention 과 같은 구성)
        self.q_proj = nn.Linear(d_model, d_model) #in, out
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)   # = W^O

    def forward(self, query, key, value, mask = None):
        """
        query: (N, Lq, d_model)   key, value: (N, Lk, d_model)
          - self-attention  : query = key = value = x
          - cross-attention : query = 디코더, key = value = 인코더 출력
        mask: (Lq, Lk) 또는 None (causal 등)
        return: (N, Lq, d_model)
        """
        N, Lq, D = query.shape
        Lk = key.shape[1]
        H, dh = self.num_heads, self.d_head

        # 투영 후 헤드로 쪼개기:
        #   (N, L, D) → (N, L, H, dh) → (N, H, L, dh)
        Q = self.q_proj(query).view(N, Lq, H, dh).transpose(1, 2)   # (N, H, Lq, dh)
        K = self.k_proj(key).view(N, Lk, H, dh).transpose(1, 2)     # (N, H, Lk, dh)
        V = self.v_proj(value).view(N, Lk, H, dh).transpose(1, 2)

        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        concat = out.transpose(1, 2).reshape(N, Lq, D)
        out = self.out_proj(concat) # 헤드 간 문맥 섞기

        return out

# 검증 
def check_sdpa():
    torch.manual_seed(0)
    N, Hh, L, dk = 2, 2, 4, 5
    Q = torch.randn(N, Hh, L, dk, dtype=torch.double)
    K = torch.randn(N, Hh, L, dk, dtype=torch.double)
    V = torch.randn(N, Hh, L, dk, dtype=torch.double)

    # 마스크 없음
    out, alpha = scaled_dot_product_attention(Q, K, V)
    ref = F.scaled_dot_product_attention(Q, K, V)
    ok1 = torch.allclose(out, ref, atol=1e-8)
    sum_ok = torch.allclose(alpha.sum(-1), torch.ones(N, Hh, L, dtype=torch.double), atol=1e-8)

    # causal 마스크 (미래 -inf)
    causal = torch.triu(torch.full((L, L), float("-inf"), dtype=torch.double), diagonal=1)
    out_c, _ = scaled_dot_product_attention(Q, K, V, mask=causal)
    ref_c = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    ok2 = torch.allclose(out_c, ref_c, atol=1e-8)

    print(f"1) SDPA        {'PASS' if (ok1 and sum_ok and ok2) else 'FAIL'}"
          f"  (no-mask {'ok' if ok1 else 'X'}, α합=1 {'ok' if sum_ok else 'X'}, causal {'ok' if ok2 else 'X'})")


def check_mha():
    torch.manual_seed(0)
    N, L, d_model, num_heads = 2, 4, 8, 2
    x = torch.randn(N, L, d_model, dtype=torch.double)

    ref = nn.MultiheadAttention(d_model, num_heads, batch_first=True, bias=True).double()
    mine = MyMultiHeadAttention(d_model, num_heads).double()

    # torch MHA weight 를 내 모듈로 복사 (in_proj_weight = [Wq; Wk; Wv])
    D = d_model
    with torch.no_grad():
        W = ref.in_proj_weight      # (3D, D)
        b = ref.in_proj_bias        # (3D,)
        mine.q_proj.weight.copy_(W[0:D]);     mine.q_proj.bias.copy_(b[0:D])
        mine.k_proj.weight.copy_(W[D:2*D]);   mine.k_proj.bias.copy_(b[D:2*D])
        mine.v_proj.weight.copy_(W[2*D:3*D]); mine.v_proj.bias.copy_(b[2*D:3*D])
        mine.out_proj.weight.copy_(ref.out_proj.weight)
        mine.out_proj.bias.copy_(ref.out_proj.bias)

    out_ref, _ = ref(x, x, x)      # self-attention: q=k=v=x
    out_mine = mine(x, x, x)
    ok = torch.allclose(out_ref, out_mine, atol=1e-8)
    print(f"2) MultiHead   {'PASS' if ok else 'FAIL'}")
    if not ok and out_mine is not None:
        print("   max diff:", (out_ref - out_mine).abs().max().item())


if __name__ == "__main__":
    check_sdpa()
    check_mha()
