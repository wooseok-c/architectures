"""
SIREN 구현, Implicit Neural Representations with Periodic Activation Functions
(Sitzmann et al., NeurIPS 2020)

  1) SineLayer.init_weights    논문 §3.2 + 부록 1.5 의 초기화
  2) SineLayer.forward         sin(omega_0 * (Wx + b))
  3) SIREN.__init__            SineLayer 여러 개 + 마지막 Linear 조립

검증: python siren.py
  - check_shapes        : 입출력 shape
  - check_init          : 논문 Theorem 1.8 재현
                          (sin 직전 = N(0,1),  sin 직후 = arcsine, var≈0.5)
                          → 층이 깊어져도 유지되는지가 핵심
  - check_derivative    : SIREN 은 2차 도함수가 살아 있고 ReLU MLP 는 0 인지

수식:
  Φ(x) = W_n ( φ_{n-1} ∘ ... ∘ φ_0 )(x) + b_n ,   φ_i(x) = sin( W_i x + b_i )   ... 식 (4)

초기화 (부록 Theorem 1.8):
  W ~ U( -sqrt(6/fan_in),  +sqrt(6/fan_in) )
  본문 §3.2 의 "c = 6" 은 오타. 부록이 맞음.

omega_0 (§3.2 + 부록 1.5):
  첫 층      sin(omega_0 * (Wx+b)) 가 [-1,1] 위에서 여러 주기를 걸치게        → 주파수
  은닉층     W = W_hat * omega_0 로 인수분해, W_hat ~ U(±sqrt(6/n)/omega_0)   → 기울기 부스트
             (순전파는 그대로, W_hat 의 기울기만 omega_0 배)
"""

import math
import torch
import torch.nn as nn

#  SineLayer  =  Linear + sin
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first          # 첫 층인가 (초기화가 다름)
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        """
        논문 §3.2 + 부록 1.5.

        첫 층  : W ~ U( -1/n,  +1/n )                      n = in_features
                 (좌표가 그대로 들어오므로 omega_0 가 주파수를 만든다)
        은닉층 : W ~ U( -sqrt(6/n)/omega_0,  +sqrt(6/n)/omega_0 )
                 (forward 에서 omega_0 를 곱하므로 실효 범위는 ±sqrt(6/n))

        bias 는 nn.Linear 기본값 그대로 둔다.
        """
        with torch.no_grad():
            # self.linear.weight.uniform_(하한, 상한) 
            # is_first 인지에 따라 범위가 다름
            #
            # 힌트: n = self.in_features
            #       math.sqrt(6 / n) / self.omega_0
            #
            pass

    def forward(self, x):
        """
        x: (..., in_features)
        return: (..., out_features)

        식 (4) 의 한 층.  주의: omega_0 는 (Wx + b) 전체에 곱한다.
        """
        # return torch.sin( ... )
        raise NotImplementedError("2) SineLayer.forward")

    def pre_activation(self, x, bias=True):
        """
        검증용. sin 을 통과하기 직전 값.

        bias=False 면 편향을 뺀 omega_0 * (Wx) 만 준다.
        논문 Theorem 1.8 이 N(0,1) 이라고 말하는 건 이쪽이다.
        증명 스케치의 괄호 "(bias does not change distribution for high
        enough frequency)" 처럼, 편향은 위상만 밀 뿐 sin 을 통과한 뒤의
        분포(arcsine)를 바꾸지 않는다. 대신 sin 직전 값의 sd 는 키운다.
        """
        z = torch.nn.functional.linear(x, self.linear.weight,
                                       self.linear.bias if bias else None)
        return self.omega_0 * z

#  SIREN 전체
class SIREN(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3,
                 out_features=3, first_omega_0=30.0, hidden_omega_0=30.0):
        """
        hidden_layers = 은닉 SineLayer 의 개수 (첫 층 제외).
        논문 이미지 fitting 은 5-layer MLP → SineLayer 4개 + 마지막 Linear 1개.

        마지막 층에는 sin 을 걸지 않는다 (출력이 [-1,1] 에 갇히면 안 되므로).
        """
        super().__init__()

        layers = []
        # 1) 첫 SineLayer:  in_features → hidden_features
        #                   is_first=True,  omega_0=first_omega_0
        # 2) 은닉 SineLayer 를 hidden_layers 개:
        #                   hidden_features → hidden_features
        #                   is_first=False, omega_0=hidden_omega_0
        # 3) 마지막 nn.Linear: hidden_features → out_features
        #    마지막 Linear 도 초기화를 맞춰주는 게 좋다:
        #       W ~ U( ±sqrt(6/hidden_features)/hidden_omega_0 )
        #
        # layers.append(...) 로 쌓으세요.

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        """coords: (N, in_features) → (N, out_features)"""
        return self.net(coords)

#  비교군: 같은 크기의 ReLU MLP
class ReLUMLP(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=3):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_features), nn.ReLU()]
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_features, hidden_features), nn.ReLU()]
        layers += [nn.Linear(hidden_features, out_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)
    
#  검증
def _todo(name, e):
    print(f"  [ ] {name}: TODO 미완성 ({type(e).__name__})")
    return False


def check_shapes():
    print("check_shapes")
    try:
        m = SIREN(in_features=2, hidden_features=64, hidden_layers=2, out_features=3)
        x = torch.rand(100, 2) * 2 - 1
        y = m(x)
    except Exception as e:
        return _todo("shape", e)

    n_sine = sum(1 for l in m.net if isinstance(l, SineLayer))
    n_lin = sum(1 for l in m.net if isinstance(l, nn.Linear))
    ok = (y.shape == (100, 3)) and n_sine == 3 and n_lin == 1
    print(f"  출력 {tuple(y.shape)} (기대 (100, 3)) | SineLayer {n_sine}개(기대 3) + Linear {n_lin}개(기대 1)")
    print("  PASS" if ok else "  FAIL")
    return ok


def check_init():
    """논문 Theorem 1.8 재현."""
    print("check_init  (Theorem 1.8: 깊이와 무관하게 분포 유지)")
    torch.manual_seed(0)
    try:
        m = SIREN(in_features=2, hidden_features=256, hidden_layers=6, out_features=1)
        x = torch.rand(4096, 2) * 2 - 1          # 정규화된 좌표 U(-1,1)
        h = x
        rows, ok = [], True
        with torch.no_grad():
            for i, layer in enumerate(m.net):
                if not isinstance(layer, SineLayer):
                    continue
                zb = layer.pre_activation(h, bias=False)   # 이론이 말하는 값
                z = layer.pre_activation(h, bias=True)     # 실제 sin 에 들어가는 값
                h = layer(h)                               # sin 직후
                rows.append((zb.std().item(), z.std().item(), h.var().item()))
    except Exception as e:
        return _todo("init", e)

    print("       층 |  Wx 만 sd  |  +bias sd  |  sin 직후 var")
    for k, (sdb, sd, var) in enumerate(rows):
        tag = "첫 층" if k == 0 else f"{k+1}층 "
        # 첫 층은 좌표가 그대로 들어오므로(omega_0 배) sd 가 1 이 아닌 게 정상
        good = True if k == 0 else (0.75 < sdb < 1.35 and 0.35 < var < 0.65)
        ok = ok and good
        print(f"     {tag} |   {sdb:7.3f}  |   {sd:7.3f}  |    {var:6.3f}   {'' if good else '  <- 벗어남'}")

    print()
    print("     기대: 2층부터  Wx 만 sd ≈ 1.0 (N(0,1))  그리고  sin 직후 var ≈ 0.5 (arcsine)")
    print("           이 둘이 층이 깊어져도 안 변하는 것 = Theorem 1.8")
    print("     '+bias' 열은 1 보다 큽니다. nn.Linear 기본 bias 에 omega_0 가 곱해져서인데,")
    print("       편향은 위상만 밀 뿐 sin 통과 후 분포는 안 바꿉니다 (그래서 var 는 0.5 유지).")
    print("       논문 Thm 1.8 증명의 괄호 '(bias does not change distribution ...)' 가 이 얘기.")
    print("     부록 1.1 은 'standard deviation of 0.5' 라고 썼지만, Lemma 1.3 의")
    print("       Var[arcsine(-1,1)] = 1/2 이므로 var 가 0.5, sd 는 0.707 입니다.")
    print("  PASS" if ok else "  FAIL  (초기화 범위를 다시 보세요)")
    return ok


def check_derivative():
    """SIREN 은 2차 도함수가 살아 있고, ReLU MLP 는 0 이어야 한다."""
    print("check_derivative  (§1 의 진단: ReLU 는 2차 미분이 0)")
    torch.manual_seed(0)

    def second_deriv(model, x):
        x = x.clone().requires_grad_(True)
        y = model(x).sum()
        g = torch.autograd.grad(y, x, create_graph=True)[0]
        g2 = torch.autograd.grad(g.sum(), x, create_graph=False)[0]
        return g.abs().mean().item(), g2.abs().mean().item()

    x = (torch.rand(512, 1) * 2 - 1)
    try:
        s = SIREN(in_features=1, hidden_features=64, hidden_layers=2, out_features=1)
        g1s, g2s = second_deriv(s, x)
    except Exception as e:
        return _todo("derivative", e)

    r = ReLUMLP(in_features=1, hidden_features=64, hidden_layers=2, out_features=1)
    g1r, g2r = second_deriv(r, x)

    print(f"     SIREN    |f'| = {g1s:.4f}   |f''| = {g2s:.4f}")
    print(f"     ReLU MLP |f'| = {g1r:.4f}   |f''| = {g2r:.4e}   <- 0 이어야 정상")
    ok = (g2s > 1e-2) and (g2r < 1e-9)
    print("  PASS" if ok else "  FAIL")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    results = [check_shapes(), check_init(), check_derivative()]
    print("=" * 60)
    print(f"{sum(bool(r) for r in results)}/{len(results)} PASS")
