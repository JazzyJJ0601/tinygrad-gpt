# tinygrad-gpt

A small GPT and the autodiff engine under it, written from scratch in NumPy so every gradient can be read and checked.
The package is called `tinygrad/` but it is its own ~700-line engine, not the [tinygrad](https://github.com/tinygrad/tinygrad) library.

## What's here

- `tinygrad/tensor.py`: a `Tensor` with reverse-mode autodiff (add, sub, mul, div, pow, matmul, sum, mean, exp, log, relu, softmax, reshape, transpose, broadcasting).
- `tinygrad/nn.py`: `Linear`, `LayerNorm`, causal self-attention, a ReLU MLP, pre-norm transformer blocks, `GPT`.
- `tinygrad/optim.py`: AdamW.
- `train.py`: trains a character-level model on `data/tiny.txt` (the opening of *Pride and Prejudice*, Project Gutenberg, 14,416 characters) and writes `results/real.json`, `loss.png` and `samples.txt`.

## Result

`python train.py` (2 layers, width 64, context 32, 108k parameters, 3,000 steps of batch 16, about 45 s on a CPU).
The text is split 80 / 10 / 10. The checkpoint with the best validation loss is kept; the test 10% is used for nothing else.

| Model | Test cross-entropy (nats/char, lower is better) |
|---|---|
| Uniform over the 73 characters | 4.290 |
| Unigram (character frequencies from train) | 3.048 |
| Bigram (previous character, from train) | 2.558 |
| **This GPT (best validation step 800)** | **2.325** |

It beats the bigram model by 0.23 nats/char, so it is using more than the previous character.
With 11.5k training characters it overfits after about 800 steps (training loss keeps falling, validation rises), which is why the best-validation checkpoint is used. A sample at temperature 1 is in `samples.txt`: English-shaped words, not sentences.

## Checked, not assumed

- `tests/test_gradcheck.py` compares every parameter's gradient in the full model with finite differences (and checks the loss equals a NumPy cross-entropy).
- `bench/compare_torch.py` compares a matmul, ReLU, sum chain with PyTorch: same loss, max gradient difference 0.

That grad check found four bugs in the first version, all fixed on 2 Oct 2026:

1. The "cross-entropy" used raw logits with no softmax, so training pushed the target logit to infinity (loss went negative and the model printed "tttt...").
2. `transpose` backward reversed every axis of a 3-D tensor instead of swapping two, and the 2-D forward did nothing; attention's query and key gradients were wrong.
3. `sum(axis)` and `mean(axis)` gradients weren't expanded to the input shape, and `mean` divided by the whole tensor's size, so LayerNorm gradients were scaled wrong.
4. The causal mask added +1e9 to the allowed positions instead of -1e9 to the future ones: equal on paper, but it destroyed float precision in the attention scores.

## Limits

- Attention is single-head: `n_heads` is accepted but not used.
- No position embeddings; order comes only from the causal mask.
- Character-level, tiny data, CPU, float64. It is for learning how the parts fit, not for real use.

## Run

```bash
pip install numpy matplotlib pytest   # torch optional, for bench/compare_torch.py
python train.py
pytest -q tests
```
