"""
Swin 의 윈도우 분할 / 복원.  둘 다 reshape 계열이라 실제 계산은 없습니다.

    (B, H, W, C)  --partition-->  (B*조개수, M*M, C)  --reverse-->  (B, H, W, C)

축 기호:
    i = 조의 행 번호 (행 // M)      a = 조 안의 행 (행 % M)
    j = 조의 열 번호 (열 // M)      b = 조 안의 열 (열 % M)

permute(0, 1, 3, 2, 4, 5) 는 2번과 3번 축만 맞바꾸는 연산이라
자기 자신이 역연산입니다. 그래서 partition 과 reverse 가 같은 숫자를 씁니다.

실행:  python window_ops.py
"""
import torch


def window_partition(x, M):
    """(B, H, W, C) -> (B * 조개수, M*M, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // M, M, W // M, M, C)      # 행·열을 각각 (조 개수, M) 으로 쪼갬
    x = x.permute(0, 1, 3, 2, 4, 5)             # B, i, j, a, b, C  로 정렬
    return x.contiguous().view(-1, M * M, C)    # 조 축끼리 / 조원 축끼리 합침


def window_reverse(windows, M, H, W):
    """(B * 조개수, M*M, C) -> (B, H, W, C)"""
    B = windows.shape[0] // ((H // M) * (W // M))
    x = windows.view(B, H // M, W // M, M, M, -1)   # 6D 로 다시 펼침
    x = x.permute(0, 1, 3, 2, 4, 5)                 # B, i, a, j, b, C  로 되돌림
    return x.contiguous().view(B, H, W, -1)         # i*a = H,  j*b = W


if __name__ == "__main__":
    # 6x6 격자, M=3  ->  조 4개 x 조원 9명
    orig = torch.arange(36).reshape(1, 6, 6, 1)
    print("입력 격자:")
    print(orig.squeeze(), "\n")

    w = window_partition(orig, M=3)
    print(f"partition 후: {tuple(w.shape)}")
    for k in range(w.shape[0]):
        print(f"  조 {k}: {w[k].flatten().tolist()}")

    back = window_reverse(w, M=3, H=6, W=6)
    print(f"\nreverse 후: {tuple(back.shape)}")
    assert torch.equal(orig, back), "왕복이 원본과 다릅니다"
    print("OK, 왕복 후 원본과 완전히 동일")

    # Swin Stage 1 실제 크기로도 확인
    big = torch.randn(2, 56, 56, 96)
    wb = window_partition(big, M=7)
    assert tuple(wb.shape) == (2 * 64, 49, 96)
    assert torch.equal(big, window_reverse(wb, M=7, H=56, W=56))
    print(f"OK, 실제 크기 (2,56,56,96) -> {tuple(wb.shape)} -> 복원 성공")
