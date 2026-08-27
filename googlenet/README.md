# GoogLeNet: Inception module

Szegedy et al., *Going Deeper with Convolutions*, CVPR 2015.

The full 22-layer network is this module stacked nine times, so one module covers the idea.

Two things it introduces:

1. **Parallel filters.** 1×1, 3×3, 5×5 and pooling branches run on the same input and their
   outputs are concatenated along the channel axis, so one layer extracts features at several
   scales at once.
2. **1×1 convolution as a bottleneck.** Reduces channels before the expensive branches. The
   same idea returns in ResNet bottlenecks and in the Transformer FFN.

## Files

| File | |
|---|---|
| `inception.py` | the module |
| `test_inception.py` | shape check |

## Running

```bash
python test_inception.py
```

Expected output: `torch.Size([2, 256, 28, 28])`.

## In the paper

Section 4 and Figure 2 cover the module. Section 5 and Table 1 give the full network.
