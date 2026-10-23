# ANG3L

A transformer language model that lives in one contract on Robinhood Chain (chain id 4663).
The weights, the tokenizer and the forward pass are all in the contract source. Ask it and it
writes you a short blessing inside a single `eth_call`: no transaction, no gas paid by anyone,
and the same words from any node.

```
cast call <ADDRESS> "bless(string,uint256)(string)" "gm" 2 \
  --rpc-url https://rpc.mainnet.chain.robinhood.com
```

`theme` is 0 builder, 1 holder, 2 night, 3 friend. Your text only seeds the sampler; the model does
not read or answer it.

## Numbers

| | |
|---|---|
| Weights | 24,064 int8, stored in `src/Weights.sol` |
| Architecture | decoder-only transformer, 2 layers, width 32, 2 heads of 16, ReLU MLP of 64, tied embeddings |
| Vocabulary / context | 203 tokens / 32 positions |
| Contract size | 33,068 bytes (the chain allows 98,304) |
| Gas per blessing | 11M to 28M measured on chain; at most about 36M for the longest output |
| Read-call limit on the public RPC | 50M gas |

## How it works

Every operation is integer math, so the contract and the Python reference agree bit for bit:

1. The words you pass are hashed: `seed = keccak256(words)`.
2. The theme token goes through the network: token and position embeddings, then per layer
   RMSNorm, attention with a key/value cache in memory, and a ReLU MLP. Weights are int8 with a
   power-of-two scale per tensor, so every rescale is an arithmetic shift. RMSNorm uses an integer
   square root, attention scales by 1/4 with a shift, softmax reads exp from a 512-entry table,
   and every division truncates toward zero, as Solidity does.
3. The next word is drawn from the top 8 logits at temperature 0.8, using
   `keccak256(seed, step)`. Ties go to the lower token id.
4. Steps 2 and 3 repeat inside the same call until the end token or 20 words.

`logits(uint256[])` exposes one raw forward pass, so any step can be checked on its own.

## Verify

```
forge test                         # the contract reproduces the reference outputs and logits
python model/live.py <ADDRESS>     # chain output vs the Python reference, with the gas each call needs
```

`test/Ang3l.t.sol` is generated from the reference: eight blessings, their token ids, and two full
logit vectors that must match exactly.

## Reproduce

```
python3.11 -m venv .venv && .venv/bin/pip install torch numpy pycryptodome
.venv/bin/python model/angel.py train     # about two minutes on a laptop CPU
.venv/bin/python model/export.py          # writes src/Weights.sol and test/Ang3l.t.sol
forge test
```

The training data is synthetic: `model/corpus.py` generates 60,000 short blessings from a small
grammar (201 distinct words). No outside text is used.

## Limits

This is a small model with a small vocabulary. It writes short, simple blessings; it does not
understand questions, and it knows nothing beyond its corpus.

## License

MIT
