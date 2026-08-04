"""
Vision Transformer (ViT) 구현 연습.
논문: An Image is Worth 16x16 Words (Dosovitskiy et al., ICLR 2021)

전체 흐름 (식 1~4):
    이미지 (B,C,H,W)
      ① 패치로 자르고 flatten     → (B, N, P²·C)       [파라미터 0]
      ② E 로 투영                → (B, N, D)          ★
      ③ CLS 를 앞에 concat        → (B, N+1, D)        ★
      ④ E_pos 를 add             → (B, N+1, D) = z₀   ★
      ⑤ 인코더 × L층              → (B, N+1, D) = z_L  ★
      ⑥ CLS(0번)만 꺼내고 LN      → (B, D) = y
      ⑦ 분류 헤드                → (B, n_classes)     ★

    z₀ = [x_class; x_p¹E; ...; x_p^N E] + E_pos          (1)
    z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}                    (2)   ← pre-norm!
    z_ℓ  = MLP(LN(z'_ℓ)) + z'_ℓ                          (3)
    y    = LN(z_L⁰)                                      (4)

막히면 print(x.shape) 를 찍기
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MyMultiHeadAttention

# ① Patch Embedding : 이미지 → 패치 토큰
class PatchEmbed(nn.Module):
    """(B, C, H, W) → (B, N, D),  N = (H/P)·(W/P)"""

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0, "img_size 는 patch_size 로 나눠떨어져야 함"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_chans     # P²·C

        ###############################################################
        # TODO A: 선형 투영 E 를 만드세요.                                 #
        #   patch_dim 을 받아 embed_dim 을 내는 nn.Linear 하나            #
        ###############################################################
        self.proj = nn.Linear(self.patch_dim, self.embed_dim) # self.porj = E matrix
        ###############################################################

    def forward(self, x):
        """x: (B, C, H, W) → (B, N, D)"""
        B, C, H, W = x.shape
        P = self.patch_size

        ###############################################################
        # TODO B: 패치로 자르고 flatten 한 뒤 self.proj 통과                #
        #                                                             #
        #   목표: (B, C, H, W) → (B, N, patch_dim) → (B, N, D)         #
        #                                                             #
        #   자르는 법 (둘 중 편한 쪽):                                     #
        #    · x.unfold(2, P, P).unfold(3, P, P) 후 permute/reshape    #
        #    · x.reshape(B, C, H//P, P, W//P, P) 후 permute/reshape    #
        #                                                             #
        #    패치 안의 (P,P,C) 가 한 덩어리로 모여야 합니다                    #
        #      패치 순서는 좌→우, 위→아래.                                  #
        ###############################################################
        x = x.reshape(B, C, H//P, P, W//P, P)
        x = x.permute(0, 2, 4, 3, 5, 1) # 세로부터 채우기
        x = x.reshape(B, self.num_patches, self.patch_dim)
        out = self.proj(x)

        return out
        # *****END OF YOUR CODE*****  

# ② ViT Encoder Block : pre-norm + GELU
class ViTBlock(nn.Module):
    """식 (2), (3).  x → x  (크기 유지)"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MyMultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        ###############################################################
        # TODO C: 식 (2), (3) 을 구현하세요.                              #
        #                                                             #
        #   z' = MSA(LN(z)) + z          ← pre-norm 주의!              #
        #   z  = MLP(LN(z')) + z'                                     #
        #                                                             #
        #   · MSA 는 self-attention (self.attn 에 q=k=v 로)            #
        #   · MLP 는 fc1 → GELU → fc2   (ReLU 아님!)                   #
        ###############################################################
        # *****START OF YOUR CODE*****
        h = self.norm1(x)
        x = x + self.attn(h, h, h)

        h = self.norm2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x

# ③ Vision Transformer 전체
class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=3, n_classes=10,
                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        N = self.patch_embed.num_patches

        ###############################################################
        # TODO D: 학습되는 파라미터 2개를 만드세요.                           #
        #                                                             #
        #   self.cls_token : CLS 토큰.  딱 1개.                         #
        #                    forward 에서 배치만큼 복제할 거라              #
        #                    (1, 1, embed_dim) 모양으로 만드세요          #
        #   self.pos_embed : 위치 임베딩. 자리 개수만큼.                    #
        #                    (1, ?, embed_dim)  — ? 는 몇일까요?         #
        #                                                             #
        #   힌트: nn.Parameter(torch.zeros(...))  로 만들고              #
        #         nn.init.trunc_normal_(p, std=0.02) 로 초기화          #
        ###############################################################
        # *****START OF YOUR CODE*****

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std = 0.02)
        # 파라미터는 배치와 무관하게 항상 1로 둔다, 알아서 복제함
        self.pos_embed = nn.Parameter(torch.zeros(1, N+1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std = 0.02)
        # E+CLS + POS

        ###############################################################

        self.blocks = nn.ModuleList(
            [ViTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """x: (B, C, H, W) → logits (B, n_classes)"""
        B = x.shape[0]

        ###############################################################
        # TODO E: 전체 파이프라인                                         #
        #                                                             #
        #   ① 패치 임베딩          self.patch_embed(x)                  #
        #   ② CLS 를 앞에 붙이기    cls_token 을 배치 B 만큼 복제한 뒤        #
        #                          torch.cat 으로 시퀀스 앞에             #
        #                          (expand 또는 repeat 사용)            #
        #   ③ pos_embed 더하기                                         #
        #   ④ self.blocks 를 순서대로 통과                               #
        #   ⑤ self.norm 적용 후 CLS 자리(0번)만 꺼내기  ← 식 (4)           #
        #   ⑥ self.head 로 분류                                        #
        ###############################################################
        # *****START OF YOUR CODE*****
        x = self.patch_embed(x)
        
        # 축 복제 B, 나머지는 그대로 -1, expand가 메모리 효율적이여서 사용
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim = 1)
        x += self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        # 최종 예측 norm, pre-norm을 했기 때문에 residual을 마지막에 norm 해주기
        x = self.norm(x)
        x = x[:, 0]

        logits = self.head(x)

        return logits
        # *****END OF YOUR CODE*****

# 검증 

def _check_patch_embed():
    torch.manual_seed(0)
    B, C, img, P, D = 2, 3, 8, 4, 16
    pe = PatchEmbed(img, P, C, D)
    x = torch.randn(B, C, img, img)
    try:
        out = pe(x)
    except NotImplementedError:
        print("① PatchEmbed   TODO 미완성"); return False

    N = (img // P) ** 2
    shape_ok = out.shape == (B, N, D)

    # 패치 내용이 올바른 위치에서 왔는지: proj 를 항등에 가깝게 두고 검사
    ref_patch = x[0, :, 0:P, 0:P].permute(1, 2, 0).reshape(-1)     # (P,P,C) 순
    ref_patch2 = x[0, :, 0:P, 0:P].reshape(-1)                      # (C,P,P) 순
    with torch.no_grad():
        w = pe.proj.weight; b = pe.proj.bias
        got = out[0, 0]
        ok_a = torch.allclose(got, w @ ref_patch + b, atol=1e-5)
        ok_b = torch.allclose(got, w @ ref_patch2 + b, atol=1e-5)
    content_ok = ok_a or ok_b

    ok = shape_ok and content_ok
    print(f"① PatchEmbed   {'PASS' if ok else 'FAIL'}"
          f"  (shape {tuple(out.shape)} {'ok' if shape_ok else f'≠ {(B,N,D)}'}"
          f", 첫 패치 내용 {'ok' if content_ok else 'X (자르는 순서 확인)'})")
    return ok


def _check_block():
    torch.manual_seed(0)
    B, L, D = 2, 5, 16
    blk = ViTBlock(D, num_heads=4)
    x = torch.randn(B, L, D)
    try:
        out = blk(x)
    except NotImplementedError:
        print("② ViTBlock     TODO 미완성"); return False

    shape_ok = out.shape == x.shape
    # residual 이 있으면 출력이 입력에서 "완전히" 벗어나지 않음
    resid_ok = (out - x).abs().mean() < x.abs().mean() * 3
    # pre-norm 확인: 입력을 크게 스케일하면 post-norm 은 출력이 거의 안 변함
    with torch.no_grad():
        big = blk(x * 50)
    prenorm_ok = (big - out).abs().mean() > 1.0

    ok = shape_ok and resid_ok and prenorm_ok
    print(f"② ViTBlock     {'PASS' if ok else 'FAIL'}"
          f"  (shape {'ok' if shape_ok else 'X'}"
          f", residual {'ok' if resid_ok else 'X'}"
          f", pre-norm {'ok' if prenorm_ok else 'X (post-norm 쓴 듯)'})")
    return ok


def _check_vit():
    torch.manual_seed(0)
    B, C, img, P, n_cls, D = 4, 3, 32, 8, 10, 64
    model = ViT(img_size=img, patch_size=P, in_chans=C, n_classes=n_cls,
                embed_dim=D, depth=2, num_heads=4)
    x = torch.randn(B, C, img, img)
    try:
        logits = model(x)
    except NotImplementedError:
        print("③ ViT          TODO 미완성"); return False

    shape_ok = logits.shape == (B, n_cls)

    # CLS/pos_embed 가 실제로 쓰였는지 (gradient 로 확인)
    logits.sum().backward()
    cls_used = model.cls_token.grad is not None and model.cls_token.grad.abs().sum() > 0
    pos_used = model.pos_embed.grad is not None and model.pos_embed.grad.abs().sum() > 0

    # 배치 내 이미지가 다르면 출력도 달라야 함
    with torch.no_grad():
        out2 = model(torch.randn(B, C, img, img))
    differs = not torch.allclose(logits, out2, atol=1e-4)

    ok = shape_ok and cls_used and pos_used and differs
    print(f"③ ViT          {'PASS' if ok else 'FAIL'}"
          f"  (출력 {tuple(logits.shape)} {'ok' if shape_ok else f'≠ {(B,n_cls)}'}"
          f", CLS 사용 {'ok' if cls_used else 'X'}"
          f", pos_embed 사용 {'ok' if pos_used else 'X'}"
          f", 입력별 출력차이 {'ok' if differs else 'X'})")
    if ok:
        n_param = sum(p.numel() for p in model.parameters())
        print(f"               파라미터 {n_param:,}개")
    return ok


if __name__ == "__main__":
    a = _check_patch_embed()
    b = _check_block()
    c = _check_vit()
    if a and b and c:
        print("\n🎉 ViT 완성!")
