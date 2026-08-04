"""
Transformer 부품들 (attention 위에 얹는 것):
  ① PositionalEncoding        위치 정보 (sin/cos)
  ② PositionwiseFeedForward   토큰별 2층 MLP
  ③ EncoderLayer              self-attn + FFN (residual + norm)
  ④ DecoderLayer              masked self-attn + cross-attn + FFN

검증: python layers.py  → 각 부품 PASS
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MyMultiHeadAttention

# ① Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # position: (max_len, 1),  div_term: (d_model/2,)  ← 여기까진 제공
        position = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))          # (d_model/2,)
        pe = torch.zeros(max_len, d_model)                              # (max_len, d_model)

        ###############################################################
        # TODO: pe 를 sin/cos 로 채우세요.                              #
        #   짝수 차원: pe[:, 0::2] = sin(position * div_term)          #
        #   홀수 차원: pe[:, 1::2] = cos(position * div_term)          #
        #   (position * div_term 이 (max_len, d_model/2) 로 broadcast)#
        ##############################################################
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)   # 학습 안 하는 고정 버퍼

    def forward(self, x):
        # x: (N, L, d_model) → 앞 L개 위치의 PE 를 더함
        return x + self.pe[:x.size(1)].unsqueeze(0)

# ② Position-wise Feed-Forward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)   # 확장 (512 → 2048)
        self.lin2 = nn.Linear(d_ff, d_model)   # 축소 (2048 → 512)

    def forward(self, x):
        ###############################################################
        # TODO: 2층 MLP.  lin1 → ReLU → lin2                          #
        #   힌트: self.lin2(F.relu(self.lin1(x)))                      #
        ###############################################################
        return self.lin2(F.relu(self.lin1(x)))

# ③ Encoder Layer  =  self-attn + FFN  (각각 residual + LayerNorm)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MyMultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """x: (N, L, d_model) → (N, L, d_model)"""
        ###############################################################
        # TODO: 서브층 2개, 각각 residual + norm.                       #
        #   패턴: out = LayerNorm(x + Sublayer(x))                     #
        #                                                             #
        #  ① self-attention (self 라 q=k=v=x):                        #
        #       x = self.norm1(x + self.self_attn(x, x, x, mask))     #
        #  ② FFN:                                                     #
        #       x = self.norm2(x + self.ffn(x))                       #
        ###############################################################
        x = self.norm1(x + self.self_attn(x, x, x, mask))
        x = self.norm2(x + self.ffn(x))

        return x

# ④ Decoder Layer  =  masked self-attn + cross-attn + FFN
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MyMultiHeadAttention(d_model, num_heads)   # masked self
        self.cross_attn = MyMultiHeadAttention(d_model, num_heads)  # 인코더 봄
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        """
        x:       (N, L_tgt, d_model)   디코더 입력
        enc_out: (N, L_src, d_model)   인코더 출력 (cross 의 key/value)
        self_mask:  causal mask (미래 가림)
        return: (N, L_tgt, d_model)
        """
        ###############################################################
        # TODO: 서브층 3개, 각각 residual + norm.                       #
        #                                                             #
        #  ① masked self-attn (q=k=v=x, self_mask):                   #
        #       x = self.norm1(x + self.self_attn(x, x, x, self_mask))#
        #  ② cross-attn (q=x, k=v=enc_out, cross_mask):               #
        #       x = self.norm2(x + self.cross_attn(x, enc_out, enc_out, cross_mask))
        #  ③ FFN:                                                     #
        #       x = self.norm3(x + self.ffn(x))                       #
        ###############################################################
        x = self.norm1(x + self.self_attn(x, x, x, self_mask))
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out, cross_mask))
        x = self.norm3(x + self.ffn(x))
        return x
# 검증 

def _check():
    torch.manual_seed(0)
    N, L, d_model, num_heads, d_ff = 2, 5, 8, 2, 32

    # ① PE
    try:
        pe = PositionalEncoding(d_model, max_len=50).double()
        pos = 3
        ok_sin = abs(pe.pe[pos, 0].item() - math.sin(pos)) < 1e-6      # div_term[0]=1
        ok_cos = abs(pe.pe[pos, 1].item() - math.cos(pos)) < 1e-6
        x = torch.randn(N, L, d_model, dtype=torch.double)
        added = pe(x)
        shape_ok = added.shape == x.shape
        print(f"① PositionalEncoding  {'PASS' if (ok_sin and ok_cos and shape_ok) else 'FAIL'}")
    except NotImplementedError:
        print("① PositionalEncoding  TODO 미완성")

    # ② FFN
    try:
        ffn = PositionwiseFeedForward(d_model, d_ff).double()
        x = torch.randn(N, L, d_model, dtype=torch.double)
        out = ffn(x)
        print(f"② FeedForward         {'PASS' if out.shape == x.shape else 'FAIL'}")
    except NotImplementedError:
        print("② FeedForward         TODO 미완성")

    # ③ EncoderLayer
    try:
        enc = EncoderLayer(d_model, num_heads, d_ff).double()
        x = torch.randn(N, L, d_model, dtype=torch.double)
        out = enc(x)
        print(f"③ EncoderLayer        {'PASS' if out.shape == x.shape else 'FAIL'}")
    except NotImplementedError:
        print("③ EncoderLayer        TODO 미완성")

    # ④ DecoderLayer
    try:
        dec = DecoderLayer(d_model, num_heads, d_ff).double()
        x = torch.randn(N, L, d_model, dtype=torch.double)
        enc_out = torch.randn(N, 6, d_model, dtype=torch.double)
        causal = torch.triu(torch.full((L, L), float("-inf"), dtype=torch.double), diagonal=1)
        out = dec(x, enc_out, self_mask=causal)
        print(f"④ DecoderLayer        {'PASS' if out.shape == x.shape else 'FAIL'}")
    except NotImplementedError:
        print("④ DecoderLayer        TODO 미완성")


if __name__ == "__main__":
    _check()
