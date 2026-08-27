"""
Transformer 전체 조립.
  1) Encoder        EncoderLayer 를 N번 쌓기
  2) Decoder        DecoderLayer 를 N번 쌓기
  3) Transformer    임베딩 + PE → 인코더 → 디코더 → 출력 투영(어휘 확률)

검증: python transformer.py  → 출력 shape (N, L_tgt, tgt_vocab) 확인 + causal mask
"""

import torch
import torch.nn as nn
import math

from layers import PositionalEncoding, EncoderLayer, DecoderLayer

def causal_mask(L, dtype=torch.float32):
    """미래를 가리는 삼각 마스크: 위치 i 는 j<=i 만 봄. (L, L), 가릴 곳 -inf."""
    return torch.triu(torch.full((L, L), float("-inf"), dtype=dtype), diagonal=1)

# 1) Encoder 스택
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(self, x, mask = None):
        for layer in self.layers: 
            x = layer(x, mask)  
        return x

# 2) Decoder 스택
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask)
        return x

# 3) Transformer 전체
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=32, num_heads=4,
                 d_ff=64, num_layers=2, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)   # 원문 토큰 → 벡터
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)   # 번역 토큰 → 벡터
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff)
        self.out_proj = nn.Linear(d_model, tgt_vocab)     # → 어휘별 점수(logits)

    def forward(self, src, tgt):
        """
        src: (N, L_src)  원문 토큰 id
        tgt: (N, L_tgt)  번역 토큰 id (지금까지)
        return: logits (N, L_tgt, tgt_vocab)  각 위치의 다음 단어 점수
        """
        enc = self.pos_enc(self.src_emb(src))
        enc_out = self.encoder(enc)

        dec = self.pos_enc(self.tgt_emb(tgt))
        mask = causal_mask(tgt.size(1), dec.dtype).to(dec.device)
        dec_out = self.decoder(dec, enc_out, self_mask = mask)

        logits = self.out_proj(dec_out)
        return logits
# 검증 
def check_transformer():
    torch.manual_seed(0)
    src_vocab, tgt_vocab = 20, 25
    N, L_src, L_tgt = 2, 7, 5

    src = torch.randint(0, src_vocab, (N, L_src))   # 원문 토큰 id
    tgt = torch.randint(0, tgt_vocab, (N, L_tgt))   # 번역 토큰 id

    try:
        model = Transformer(src_vocab, tgt_vocab, d_model=32, num_heads=4,
                            d_ff=64, num_layers=2, max_len=100)
        logits = model(src, tgt)
    except NotImplementedError as e:
        print("Transformer  미구현:", e)
        return

    shape_ok = logits.shape == (N, L_tgt, tgt_vocab)
    # gradient 흐르는지 (전체가 미분가능하게 연결됐는지)
    loss = logits.sum()
    loss.backward()
    grad_ok = model.src_emb.weight.grad is not None

    ok = shape_ok and grad_ok
    print(f"Transformer  {'PASS' if ok else 'FAIL'}"
          f"  (출력 shape {'ok' if shape_ok else 'X'} = {tuple(logits.shape)}, "
          f"gradient {'ok' if grad_ok else 'X'})")


if __name__ == "__main__":
    check_transformer()
