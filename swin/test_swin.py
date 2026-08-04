"""
Swin-T 검증 — 부품 하나씩 확인.
실행:  python test_swin.py

아직 안 만든 부품은 자동으로 건너뜁니다. 위에서부터 하나씩 초록불을 켜세요.
"""
import torch
import torch.nn as nn

import swin


def run(name, fn):
    try:
        fn()
        print(f"✅ {name}")
    except NotImplementedError:
        print(f"⬜ {name}  — 아직 구현 안 함")
    except AssertionError as e:
        print(f"❌ {name}  — {e}")
    except Exception as e:
        print(f"💥 {name}  — {type(e).__name__}: {e}")


def eq(got, want, what="shape"):
    assert tuple(got) == tuple(want), f"{what} 이 {tuple(got)}, 기대값은 {tuple(want)}"


# ── 1) Mlp ────────────────────────────────────────────────────
def t_mlp():
    m = swin.Mlp(96, mlp_ratio=4.0)
    eq(m(torch.randn(2, 49, 96)).shape, (2, 49, 96))
    # 은닉층이 4배인지 (96 -> 384 -> 96):  파라미터 수로 확인
    n = sum(p.numel() for p in m.parameters())
    assert n == 96 * 384 + 384 + 384 * 96 + 96, f"파라미터 {n}개 — 은닉 384 가 맞는지 확인"


# ── 2) PatchEmbed ─────────────────────────────────────────────
def t_patch_embed():
    pe = swin.PatchEmbed(3, 96, patch_size=4)
    eq(pe(torch.randn(2, 3, 224, 224)).shape, (2, 56, 56, 96))
    # 채널이 맨 뒤로 갔는지 (permute 를 빠뜨리면 (2, 96, 56, 56) 이 나옵니다)


# ── 3) PatchMerging ───────────────────────────────────────────
def t_patch_merging():
    pm = swin.PatchMerging(96)
    eq(pm(torch.randn(2, 56, 56, 96)).shape, (2, 28, 28, 192))

    # 정말로 "이웃한 2x2" 를 묶는지 확인.
    # norm 과 reduction 을 Identity 로 갈아끼워 concat 결과를 직접 들여다본다.
    pm2 = swin.PatchMerging(1)
    pm2.norm = nn.Identity()
    pm2.reduction = nn.Identity()

    #  0  1  2  3        왼쪽 위 2x2 묶음 = 0, 1, 4, 5
    #  4  5  6  7        오른쪽 위 2x2    = 2, 3, 6, 7
    #  8  9 10 11
    # 12 13 14 15
    x = torch.arange(16, dtype=torch.float).reshape(1, 4, 4, 1)
    out = pm2(x)                                 # (1, 2, 2, 4)
    eq(out.shape, (1, 2, 2, 4), "concat 결과 shape")

    got = out[0, 0, 0].tolist()                  # (0,0) 자리에 모인 네 값
    want = [0., 4., 1., 5.]                      # x0, x1, x2, x3 = 왼위/왼아래/오른위/오른아래
    assert got == want, (
        f"(0,0) 자리에 {got} 가 모였습니다. 기대값은 {want} "
        f"(원래 격자의 왼쪽 위 2x2 = 0,1,4,5) — 슬라이싱이 이웃을 안 묶고 있습니다")


# ── 4) WindowAttention ────────────────────────────────────────
def t_window_attention():
    M, C, h = 7, 96, 3
    attn = swin.WindowAttention(C, M, h)
    x = torch.randn(64 * 2, M * M, C)            # 조 64개 x 배치 2
    eq(attn(x).shape, (128, 49, 96))

    # relative position index 검산: 대각선은 전부 같은 번호여야 함
    idx = attn.relative_position_index
    eq(idx.shape, (49, 49), "index shape")
    d = torch.diagonal(idx)
    assert (d == d[0]).all(), "대각선(자기 자신)의 인덱스가 서로 다릅니다"

    # bias 표 크기
    eq(attn.relative_position_bias_table.shape, ((2 * M - 1) ** 2, h), "bias table")

    # mask 를 주면 결과가 달라져야 함
    mask = torch.zeros(64, M * M, M * M)
    mask[0, 0, 1:] = -100.0
    assert not torch.allclose(attn(x), attn(x, mask=mask)), "mask 가 반영되지 않았습니다"


# ── 5) SwinBlock ──────────────────────────────────────────────
def t_swin_block():
    # W-MSA (shift 0)
    b0 = swin.SwinBlock(96, (56, 56), num_heads=3, window_size=7, shift_size=0)
    eq(b0(torch.randn(2, 56, 56, 96)).shape, (2, 56, 56, 96))
    assert b0.attn_mask is None, "W-MSA 인데 mask 가 만들어졌습니다"

    # SW-MSA (shift 3)
    b1 = swin.SwinBlock(96, (56, 56), num_heads=3, window_size=7, shift_size=3)
    eq(b1(torch.randn(2, 56, 56, 96)).shape, (2, 56, 56, 96))
    assert b1.attn_mask is not None, "SW-MSA 인데 mask 가 없습니다"
    eq(b1.attn_mask.shape, (64, 49, 49), "mask shape")
    vals = set(b1.attn_mask.unique().tolist())
    assert vals <= {0.0, -100.0}, f"mask 값이 {vals} — 0 과 -100 만 있어야 합니다"

    # Stage 4 (7x7 격자, M=7): 나눌 게 없으니 shift 가 꺼져야 함
    b2 = swin.SwinBlock(768, (7, 7), num_heads=24, window_size=7, shift_size=3)
    assert b2.shift_size == 0, "격자와 윈도우가 같으면 shift 를 꺼야 합니다"
    eq(b2(torch.randn(2, 7, 7, 768)).shape, (2, 7, 7, 768))


# ── 6) SwinTransformer 전체 ───────────────────────────────────
def t_full():
    model = swin.SwinTransformer()
    eq(model(torch.randn(2, 3, 224, 224)).shape, (2, 1000))

    # stage 별 출력 해상도 — FPN 이 기대하는 규격
    feats = model.forward_features(torch.randn(1, 3, 224, 224))
    assert len(feats) == 4, f"stage 출력이 {len(feats)}개 — 4개여야 합니다"
    for f, want in zip(feats, [(1, 56, 56, 96), (1, 28, 28, 192),
                               (1, 14, 14, 384), (1, 7, 7, 768)]):
        eq(f.shape, want, "stage 출력")

    # 파라미터 수 — 논문 Table 1 의 Swin-T 는 29M
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"     파라미터 {n:.1f}M   (논문 Table 1: 29M)")
    assert 26 < n < 32, f"{n:.1f}M — 어딘가 차원이 어긋난 것 같습니다"


if __name__ == "__main__":
    print("=" * 55)
    run("1) Mlp", t_mlp)
    run("2) PatchEmbed", t_patch_embed)
    run("3) PatchMerging", t_patch_merging)
    run("4) WindowAttention", t_window_attention)
    run("5) SwinBlock", t_swin_block)
    run("6) SwinTransformer", t_full)
    print("=" * 55)
