"""
Swin Transformer (Swin-T) 구현 

논문:  Liu et al., Swin Transformer, ICCV 2021
이미 완성:  window_ops.py  (window_partition / window_reverse)
────────────────────────────────────────────────────────────────
텐서 규약

    이 파일에서는 격자 형태 (B, H, W, C) 
      B = 배치,  H·W = 격자,  C = 채널

    윈도우로 자른 뒤에는 (B*조개수, M*M, C).
      조 개수 = (H//M) * (W//M),  M*M = 조원 수

    ※ 공식 구현은 (B, L, C) 로 평탄화해서 다니는데, 여기서는 격자 모양을
      그대로 유지합니다. window_ops.py 와 맞고 shape 을 눈으로 따라가기 쉬워서
────────────────────────────────────────────────────────────────
전체 흐름 (Swin-T, 224×224, C=96)

    (B, 3, 224, 224)
       │ PatchEmbed          4×4 픽셀 -> 토큰 1개
       ▼
    (B, 56, 56, 96)
       │ Stage 1  블록 ×2   (W-MSA, SW-MSA)
       │ PatchMerging
       ▼
    (B, 28, 28, 192)
       │ Stage 2  블록 ×2
       │ PatchMerging
       ▼
    (B, 14, 14, 384)
       │ Stage 3  블록 ×6
       │ PatchMerging
       ▼
    (B, 7, 7, 768)
       │ Stage 4  블록 ×2      조가 1개 = 전역 attention
       ▼
    LN -> mean -> Linear(768, 1000) -> logits

────────────────────────────────────────────────────────────────
    1) Mlp            
    2) PatchEmbed
    3) PatchMerging
    4) WindowAttention (핵심)
    5) SwinBlock
    6) SwinTransformer
"""
import torch
import torch.nn as nn

from window_ops import window_partition, window_reverse

# 1) Mlp — 블록 안의 feed-forward 부분
class Mlp(nn.Module):
    """
    Linear -> GELU -> Linear
    Transformer 표준 부품이라 Swin 특유의 것은 없음.

        (..., dim) -> (..., dim*4) -> (..., dim)

    hidden 을 4배로 부풀렸다 되돌리는 게 관례 (논문의 MLP 확장률 alpha=4)
    입력 차원과 출력 차원이 같아서 residual 로 더할 수 있다
    """

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim ,dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# 2) PatchEmbed — 이미지를 토큰 격자로
class PatchEmbed(nn.Module):
    """
        (B, 3, 224, 224)  ->  (B, 56, 56, 96)

    논문 §3.1 

        Patch Partition   4×4 픽셀을 묶어 토큰 1개로   -> 토큰당 4*4*3 = 48개 숫자
        Linear Embedding  48 -> 96 으로 사영

    구현에서는 Conv2d 사용해서 한번에 연산
        nn.Conv2d(3, 96, kernel_size=4, stride=4)

    왜 같은가:
      - kernel 4×4, stride 4  =  겹치지 않게 4×4 픽셀씩 훑음  (= Patch Partition)
      - 출력 채널 96          =  그 48개 값을 96개로 사영     (= Linear Embedding)
      conv 로 쓰면 자르기+곱하기를 한 번에 하니 빠릅니다.

    주의: Conv2d 는 (B, C, H, W)인데 여기서는 (B, H, W, C) 
          permute 사용해서 조작하기
    """

    def __init__(self, in_chans=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size,  stride = patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

# 3) PatchMerging — 2×2를 하나로
class PatchMerging(nn.Module):
    """
        (B, H, W, C)  ->  (B, H/2, W/2, 2C)
    예시:  (B, 56, 56, 96)  ->  (B, 28, 28, 192)
    2단계로 구성
        1) concat : 이웃한 2×2 네 명의 벡터를 이어붙임    C -> 4C   (96 -> 384)
                    자리만 합치는 거라 정보 보존
        2) Linear : 너무 두꺼우니 절반으로 압축           4C -> 2C  (384 -> 192)
                    안 줄이면 96 -> 384 -> 1536 -> 6144 폭발

    결과: 학생 1/4, 채널 2배, 총 데이터량 절반

        x0 = x[:, 0::2, 0::2, :]     짝수 행, 짝수 열   (왼쪽 위)
        x1 = x[:, 1::2, 0::2, :]     홀수 행, 짝수 열   (왼쪽 아래)
        x2 = x[:, 0::2, 1::2, :]     짝수 행, 홀수 열   (오른쪽 위)
        x3 = x[:, 1::2, 1::2, :]     홀수 행, 홀수 열   (오른쪽 아래)

    `0::2` 는 "0번부터 2칸씩" 이라는 뜻입니다. 각각 (B, H/2, W/2, C) 가 되고,
    같은 자리끼리 모으면 원래 격자의 2×2 묶음 하나

    격자로 보면 숫자가 몇 번째 x 에 뽑히는지:

           열0 열1 열2 열3
        행0  0   2   0   2
        행1  1   3   1   3
        행2  0   2   0   2
        행3  1   3   1   3

    공식 구현의 concat 순서가 x0, x1, x2, x3 (위/아래/위/아래) 
    순서 자체는 뒤의 Linear 가 학습으로 맞추므로 무엇이든 상관없지만 일관성 유지
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias = False)  

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"격자가 홀수입니다: {H}x{W}"

        x0 = x[:, 0::2, 0::2, ]
        x1 = x[:, 1::2, 0::2, ]
        x2 = x[:, 0::2, 1::2, ]
        x3 = x[:, 1::2, 1::2, ]
        x = torch.cat([x0, x1, x2, x3], dim = -1)
        x = self.norm(x)
        x = self.reduction(x)

        return x
    
# 4) WindowAttention, 조 안에서의 attention  (핵심)
class WindowAttention(nn.Module):
    """
        (조 개수*B, M*M, C)  ->  (조 개수*B, M*M, C)

    조 하나 안에서만 도는 평범한 multi-head self-attention 
    ViT 와 다른 점은 딱 두 가지:
        1. relative position bias 를 점수에 더한다
        2. SW-MSA 일 때 mask 를 더한다

    relative position bias
    학습하는 표는 (2M-1)^2 개.  M=7 이면 13*13 = 169개 (head 마다).
    채워야 할 자리는 M*M x M*M = 49*49 = 2,401칸.

    같은 상대 위치를 가진 쌍끼리 값을 공유하므로, "어느 쌍이 표의 몇 번째를
    볼지" 를 미리 계산해두고 인덱싱 한 번으로 끝

        bias = table[index]  index: (49, 49),  table: (169, num_heads)

    index 만드는 과정 
        1) 조원 49명의 (행, 열) 좌표를 만든다
        2) 모든 쌍의 상대 좌표를 구한다        범위 -6 ~ +6
        3) M-1 을 더해 음수를 없앤다            범위  0 ~ 12
        4) 2D 좌표를 1D 번호로 편다             행*(2M-1) + 열
    검산: 대각선(자기 자신과의 쌍)은 상대 좌표가 (0,0) 이라 전부 같은 번호

    mask
    W-MSA 는 mask=None.  SW-MSA 일 때만 (조 개수, M*M, M*M) 짜리가 들어옴
    0 = 허용, -100 = 차단.  softmax 전에 더하면 차단된 자리는 가중치가 0이 됨
    """

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.M = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads          # Swin 은 항상 32
        self.scale = head_dim ** -0.5        # 1/sqrt(d)

        # 학습되는 bias 표: (2M-1)^2 개 x head 수
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 인덱스 미리 계산 (학습 대상 아님, 위치만의 함수) 
        M = window_size
        coords = torch.stack(torch.meshgrid(
            torch.arange(M), torch.arange(M), indexing="ij"))   # (2, M, M)
        coords = torch.flatten(coords, 1)                       # (2, M*M)
        rel = coords[:, :, None] - coords[:, None, :]           # None은 Unsqueeze와 같음
        rel = rel.permute(1, 2, 0).contiguous()                 # (M*M, M*M, 2)
        rel[:, :, 0] += M - 1                                   # 음수 제거: -6~6  ->  0~12
        rel[:, :, 1] += M - 1
        rel[:, :, 0] *= 2 * M - 1                               # 2D -> 1D:  행*(2M-1) + 열
        self.register_buffer("relative_position_index", rel.sum(-1))   # (M*M, M*M)

        
        self.qkv = nn.Linear(dim, 3 * dim, bias = True)
        self.proj = nn.Linear(dim ,dim)

    def forward(self, x, mask=None):
        """
        x    : (B_, N, C)              B_ = 조개수*B,  N = M*M
        mask : (조개수, N, N) 또는 None
        """
        B_, N, C = x.shape

        # (1) Q, K, V 만들기 
        # (B_, N, 3C)
        # (B_, N, 3, h, d)
        # (3, B_, h, N, d)
        # 각각 (B_, h, N, d)

        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        #   왜 이렇게? attention 은 head 마다 독립이므로 head 축(h)을
        #   앞쪽 배치 자리로 보내야 뒤의 행렬곱이 head 별 계산
        
        # (2) attention 점수 
        #   q = q * self.scale
        #   attn = q @ k.transpose(-2, -1)                 -> (B_, h, N, N)
        q = q * self.scale # sqrt(d_K) 값을 미리 곱해줌, 수학적으로 동일
        attn = q @ k.transpose(-2, -1)

        # (3) relative position bias 더하기 
        #   idx  = self.relative_position_index.view(-1)              (N*N,)
        #   bias = self.relative_position_bias_table[idx]             (N*N, h)
        #   bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()  (h, N, N)
        #   attn = attn + bias.unsqueeze(0)                           브로드캐스팅
        idx = self.relative_position_index.view(-1)
        bias = self.relative_position_bias_table[idx]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0) # (128, 3, 49, 49) + bias

        # (4) mask 더하기 (SW-MSA 일 때만) -> shape 맞추는 부분
        #   if mask is not None:
        #       nW = mask.shape[0]                                    조 개수
        #       attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
        #       attn = attn + mask.unsqueeze(1).unsqueeze(0)          (1, nW, 1, N, N)
        #       attn = attn.view(-1, self.num_heads, N, N)
        #   조 개수 축을 잠깐 꺼내야 조마다 다른 mask 를 더할 수 있다.
        #   unsqueeze 두 번은 배치 축과 head 축 자리를 비워두는 것 
        if mask is not None:
            nW = mask.shape[0] # number of Windows
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) # 8조를 2 * 4로 정리
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        # (5) softmax -> V 곱하기 -> head 합치기 
        #   attn = attn.softmax(dim=-1)
        #   x = (attn @ v)                       -> (B_, h, N, d)
        #   x = x.transpose(1, 2).reshape(B_, N, C)   head 를 다시 붙임
        #   x = self.proj(x)
        attn = attn.softmax(dim = -1)
        x = attn @ v
        x = x.transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        return x
# 5) SwinBlock — 블록 하나
class SwinBlock(nn.Module):
    """
        (B, H, W, C)  ->  (B, H, W, C)   격자 크기를 바꾸는 것이 아님

    구조 (pre-norm):

        z ─────────────────────┐
        LN                     │
        (S)W-MSA               │ residual
        + ←────────────────────┘
        │
        ├──────────────────────┐
        LN                     │
        MLP                    │ residual
        + ←────────────────────┘

    shift_size = 0 이면 W-MSA, M//2 이면 SW-MSA 
    블록을 쌓을 때 0, M//2, 0, M//2 ... 로 번갈

    ── SW-MSA 의 추가 절차 
        roll(-s, -s)  ->  window_partition  ->  attn(mask)
                      ->  window_reverse    ->  roll(+s, +s)

    ── mask 만들기 
    굴린 격자에서 "반대편 끝에서 굴러온" 애들이 같은 조에 섞입니다.
    그 쌍만 막아야 합니다. 방법:

        1) 격자를 9개 지역으로 나눠 번호를 칠한다 (3덩이 x 3덩이)
             slices = (slice(0, -M), slice(-M, -s), slice(-s, None))
             - 앞부분: 완전히 정상인 조들
             - 중간  : 마지막 조의 진짜 부분
             - 끝    : 굴러 넘어온 부분
        2) 같은 window_partition 으로 자른다
        3) 조 안에서 번호끼리 뺀다  ->  0 이면 같은 지역
        4) 0 이 아닌 자리에 -100

    mask 는 위치에만 의존하므로 __init__ 에서 한 번 만들어 buffer 에 저장합니다.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.H, self.W = input_resolution
        self.M = window_size
        self.shift_size = shift_size

        # 격자가 윈도우보다 작으면 (Stage 4: 7x7, M=7) 나눌 게 없으므로
        # 윈도우를 격자 크기로 줄이고 shift 끔 -> 자동으로 전역 attention
        if min(self.H, self.W) <= self.M:
            self.M = min(self.H, self.W)
            self.shift_size = 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.M, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

        self.register_buffer("attn_mask", self._build_mask())

    def _build_mask(self):
        """SW-MSA 용 attention mask.  W-MSA 면 None."""
        if self.shift_size == 0:
            return None

        H, W, M, s = self.H, self.W, self.M, self.shift_size

        img_mask = torch.zeros((1, H, W, 1))
        slices = (slice(0, -M), slice(-M, -s), slice(-s, None))
        cnt = 0
        for a in slices:
            for b in slices:
                img_mask[:, a, b, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, M).view(-1, M * M)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        
        return attn_mask.masked_fill(attn_mask != 0, -100.0)
        
    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        M, s = self.M, self.shift_size

        shortcut = x
        x = self.norm1(x)

        if s > 0:
            x = torch.roll(x, shifts = (-s, -s), dims = (1,2))

        x = window_partition(x, M)
        x = self.attn(x, mask = self.attn_mask)
        x = window_reverse(x, M, H, W)

        if s > 0:
            x = torch.roll(x, shifts = (s, s), dims =(1,2))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
    
# 6) SwinTransformer — 전체 조립
class SwinTransformer(nn.Module):
    """
        (B, 3, 224, 224)  ->  (B, num_classes)

    Swin-T 기본값:
        embed_dim 96,  depths (2,2,6,2),  num_heads (3,6,12,24),  window 7
    Stage i 의 격자는 224/4 / 2^i,  채널은 96 * 2^i:

        Stage 1:  56x56,  96      head 3
        Stage 2:  28x28,  192     head 6
        Stage 3:  14x14,  384     head 12
        Stage 4:   7x7,   768     head 24

    블록의 shift 는 0, M//2, 0, M//2 ... 로 번갈아 나옴
    depths 가 전부 짝수인 이유 (W 와 SW 가 짝을 이뤄야 함)

    PatchMerging 은 stage 사이에 들어가므로 마지막 stage 뒤에는 없음
    """

    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.num_stages = len(depths)
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size=4)

        
        # 마지막 차원 = embed_dim * 2**(num_stages-1) = 768
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        
        for i in range(self.num_stages):
            dim = embed_dim * 2 ** i
            res = img_size // 4 // 2 ** i

            blocks = nn.ModuleList()
            for j in range(depths[i]):
                shift = 0 if j % 2 == 0 else window_size // 2
                blocks.append(SwinBlock(dim, (res, res), num_heads[i],
                              window_size, shift, mlp_ratio))
                
            self.stages.append(blocks)

            # 마지막 stage 뒤에는 merging 이 없다 (다음이 분류기라 더 줄일 필요 없음)
            if i < self.num_stages - 1:
                self.merges.append(PatchMerging(dim))

        self.norm = nn.LayerNorm(embed_dim * 2 ** (self.num_stages - 1))
        self.head = nn.Linear(embed_dim * 2 ** (self.num_stages - 1), num_classes)

    def forward_features(self, x):
        """(B, 3, 224, 224) -> stage 별 출력 리스트 (검출용 backbone 으로 쓸 때 필요)"""
        
        x = self.patch_embed(x)
        outs = []
        for i , blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x)
            outs.append(x)              # merge 전에 저장해야 그 stage 의 해상도가 남는다
            if i < len(self.merges):
                x = self.merges[i](x)

        return outs                     

    def forward(self, x):
        feats = self.forward_features(x)
        x = feats[-1]                   # 마지막 stage 만 사용   (B, 7, 7, 768)
        x = self.norm(x)
        x = x.flatten(1, 2).mean(dim=1) # 격자를 펴서 49명 평균   -> (B, 768)
        return self.head(x)             # -> (B, num_classes)
        
