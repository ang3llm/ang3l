# 001 · The packed int8 kernel

**Question.** A transformer is mostly matrix-vector products, so the gas the EVM spends on one multiply-accumulate
(MAC) decides how large a model can answer inside a single `eth_call`. How low can it go, exactly?

## First idea: twelve products in one multiply

[`ref.py`](ref.py) packs twelve unsigned bytes into one 256-bit word at 20-bit spacing, and twelve more into a
second word in reverse order. Bits 220 to 239 of their product then hold the dot product of the twelve pairs. The
script checks it on 200,000 random cases and on the worst case. In Solidity, with the packing and unpacking around
it, this first approach came to about 15 gas per MAC: too slow for a model of a million weights in one call.

## The kernel AM3N runs

[`Kernel.sol`](Kernel.sol) turns the problem around. It packs the weights once, when they are written, and spends
one EVM `MUL` on eight of them:

- Weights are stored as unsigned bytes, offset by +128. A block of 32 output rows keeps each input column's 32
  weights in one 256-bit word; row `8p + L` sits at byte `31 − 4L − p`.
- A mask keeps one byte of every 32-bit lane, so one `MUL` by the activation scales eight weights at once into
  eight separate lanes. Four shifted masks cover the whole word.
- Activations are shifted beforehand so that no lane can overflow: `|x| ≤ 2^(23 − log2 n_in)`.
- After the loop, one correction per row (`−128 · Σx`) removes the offset, and the signed lanes are peeled.

[`KernelTest.sol`](KernelTest.sol) multiplies a 256 × 256 int8 matrix by a vector and checks every output against
the plain loop:

| | |
|---|---|
| Result | exact, all 256 outputs |
| Gas for the 256 × 256 product | 404,725 |
| Gas per multiply-accumulate | **6.17** |

That number is what lets AM3N exist. With top-1 routing a word touches 319,744 of the model's 1,239,424 weights, and
a whole blessing of up to ten tokens costs 29M to 41M gas, under the 50M a public node allows for one read.

## Run it

```
python3 ref.py
forge test --root research/001-packed-kernel -vv
```
