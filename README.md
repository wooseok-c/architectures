# architectures

Published deep learning architectures, re-implemented in PyTorch from the papers, module by
module. Each directory has a test that checks shapes end to end.

These are not models I designed. I wrote them to study them, without a reference implementation
open, so the code is written for reading rather than for speed.

| Directory | Architecture | Paper |
|---|---|---|
| [`swin/`](swin) | Swin Transformer (Swin-T) | Liu et al., ICCV 2021 |
| [`transformer/`](transformer) | Transformer, ViT | Vaswani et al. 2017 · Dosovitskiy et al. 2021 |
| [`rnn/`](rnn) | RNN, GRU/LSTM, Seq2Seq, Attention | Hochreiter & Schmidhuber · Cho · Sutskever · Bahdanau |
| [`resnet/`](resnet) | ResNet BasicBlock | He et al., CVPR 2016 |
| [`googlenet/`](googlenet) | Inception module | Szegedy et al., CVPR 2015 |

## Notes

- **Swin** is the only full model here rather than a single block. It comes to 28.3M parameters;
  the paper's Table 1 reports 29M. Tensors stay in grid form `(B, H, W, C)` instead of the
  official flattened `(B, L, C)`, which costs a few reshapes but makes the window shift easier
  to follow. Includes the relative position bias table and the shift mask.
- **Recurrent models** are built in the order RNN cell, sequence layer, seq2seq, so the
  context-vector bottleneck appears before attention is added. GRU and LSTM gates are written
  out rather than delegated to `nn.LSTM`.
- **ViT** is at CIFAR scale (32×32, 10 classes), not the ImageNet configuration.

Longer write-up, in Korean: [`swin/Swin_정리.md`](swin/Swin_정리.md)

## Running

Requires `torch`.

```bash
cd swin && python test_swin.py
```

Each directory has its own `test_*.py`.

---

Woo-Seok Choi · [wooseok-c.github.io](https://wooseok-c.github.io)
