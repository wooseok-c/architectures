# RNN, GRU/LSTM, Seq2Seq, Attention

Forward structure only, no training. The point is to see how a sequence is carried through the
network and where that breaks down.

The cell is two affine maps and a `tanh`, with the same weights reused at every step:

```
h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)
```

Built bottom up, in this order:

| File | |
|---|---|
| `rnn_cell.py` | one step |
| `rnn_layer.py` | the sequence, looping the cell and passing `h` along |
| `gru.py`, `lstm.py` | gates written out individually |
| `seq2seq.py` | encoder and decoder |
| `attention.py` | additive attention over the encoder states |
| `test_rnn.py` | shape checks for cell, layer and seq2seq |

The order matters. The encoder's last `h` is a single vector standing in for the whole input,
which is the bottleneck; the decoder generates from it one token at a time, feeding its own
output back. Attention exists to remove that bottleneck by letting the decoder read every
encoder state directly, so it is easier to see the point of after writing the version that
lacks it.

## Running

```bash
python test_rnn.py
```
