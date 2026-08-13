# ResNet BasicBlock

He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.

A layer learns the residual `F(x) = H(x) - x` instead of the whole mapping, and the input is
added back through a shortcut:

```
y = F(x) + x
```

If the identity is what the layer needs, `F` only has to go to zero, which is what makes very
deep networks trainable. The `+ x` here is the same Add that later appears in the Transformer's
Add & Norm.

## sum vs concat

Inception concatenates, so its branches only need matching `H·W`. ResNet adds, so channels and
spatial size must both match. That constraint is the reason the projection shortcut exists.

## Three shortcut cases

| Case | Shortcut |
|---|---|
| `in == out`, stride 1 | identity: `y = F(x) + x` |
| channels differ | 1×1 conv, stride 1: `y = F(x) + Ws·x` |
| channels and size differ | 1×1 conv, stride 2 |

## Files

| File | |
|---|---|
| `basic_block.py` | the block |
| `test_basic_block.py` | shape check over all three cases |

## Running

```bash
python test_basic_block.py
```

## In the paper

Section 1 states the degradation problem. Sections 3.1 and 3.2 with Figure 2 cover the residual
block.
