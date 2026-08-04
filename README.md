# architectures

Deep learning architectures rebuilt in PyTorch from the original papers — module by module,
each with a test that checks shapes end to end.

The point is not to have a fast implementation; it is to have understood the paper well enough
to write the layer. Where a reference implementation takes a shortcut that hides the idea,
I took the longer route on purpose (see the Swin note below).

| Directory | Architecture | Paper | Check |
|---|---|---|---|
| [`swin/`](swin) | Swin Transformer (Swin-T) | Liu et al., ICCV 2021 | **28.3M params** vs 29M in paper Table 1 · 6/6 components |
| [`transformer/`](transformer) | Transformer, ViT | Vaswani et al. 2017 · Dosovitskiy et al. 2021 | forward passes verified |
| [`rnn/`](rnn) | RNN → GRU/LSTM → Seq2Seq → Attention | Hochreiter & Schmidhuber · Cho · Sutskever · Bahdanau | cell / layer / seq2seq shapes |
| [`resnet/`](resnet) | ResNet BasicBlock | He et al., CVPR 2016 | 3 shortcut cases |
| [`googlenet/`](googlenet) | Inception module | Szegedy et al., CVPR 2015 | `(2, 256, 28, 28)` |

## Notes

- **Swin** — tensors are kept in grid form `(B, H, W, C)` rather than the official flattened
  `(B, L, C)`. It costs a few reshapes, but the shifted window is something you can follow by
  hand instead of taking on faith. Includes the relative position bias table and the attention
  mask that makes the cyclic shift correct.
- **Recurrent models** — built in the order RNN cell → sequence layer → seq2seq, so the
  context-vector bottleneck shows up on its own before attention is introduced to fix it.
  GRU and LSTM gates are written out rather than delegated to `nn.LSTM`.
- **ViT** is at CIFAR scale (32×32, 10 classes), not the ImageNet configuration.

Longer write-up (in Korean): [`swin/Swin_정리.md`](swin/Swin_정리.md)

## Running

Requires `torch`.

```bash
cd swin && python test_swin.py
```

Each directory has its own `test_*.py`.

---

Woo-Seok Choi · [wooseok-c.github.io](https://wooseok-c.github.io)
