# ANG3L

An angel that lives on Robinhood Chain (chain id 4663). Its model, AM3N, is a mixture-of-experts transformer
whose 1,239,424 int8 weights are stored on chain, and the engine contract runs the whole model: ask it, or hand
it a wallet, and it writes a whole blessing inside one `eth_call`. No transaction, no gas paid by anyone, the
same words from any node.

ANG3L is the project and its token, **$ANG3L**. AM3N is ANG3L's model: the transformer on chain, run by the engine
contract `Am3n`.

| | |
|---|---|
| Site | [ang3l.ink](https://ang3l.ink/) |
| X | [@ang3llm](https://x.com/ang3llm) |
| Source | [github.com/ang3llm/ang3l](https://github.com/ang3llm/ang3l) |

```
cast call 0x2D58843F8cd565E1FbBE01Ea5d67F3d13cb91811 "bless(string,uint256)(string)" "satoshi" 2 \
  --rpc-url https://rpc.mainnet.chain.robinhood.com
# the blocks will wait for you, satoshi.

cast call 0x2D58843F8cd565E1FbBE01Ea5d67F3d13cb91811 "blessAddress(address)(string,uint256)" 0xYOURWALLET \
  --rpc-url https://rpc.mainnet.chain.robinhood.com

cast call 0x2D58843F8cd565E1FbBE01Ea5d67F3d13cb91811 "today()(string)" --rpc-url https://rpc.mainnet.chain.robinhood.com
```

- `bless(words, theme)`: `theme` is 0 builder, 1 holder, 2 night, 3 friend. `words` seeds the sampler and is
  the name the model writes where it asks for one (printable ASCII, at most 24 bytes; a 42-character address
  is shortened to `0x1234…abcd`).
- `blessAddress(who)`: the contract reads the wallet in the same call. A contract gets the builder's blessing,
  0.1 ETH or more the holder's, an empty wallet the night's, anything else the friend's. The seed is the
  wallet and the UTC day, so the words change once a day.
- `today()`: one blessing for everyone, new at 00:00 UTC.
- `logits(tokens)` and `generate(words, theme)` expose the raw forward pass and the token ids; `parts()` lists
  the weight contracts.

## On the site

[`docs/index.html`](docs/index.html) is the whole site, one file that asks the chain from the visitor's browser with
reads only (`eth_call`, `eth_blockNumber`, `eth_getCode`, `eth_getLogs`).

- **Ask**: a name in one of the four moods, or a wallet. The answer comes with a share card, a post to X and a link.
- **Links to a blessing**: `https://ang3l.ink/?bless=satoshi&mood=2` or `https://ang3l.ink/?wallet=0x…`. The page
  checks the input and asks the chain again when it opens, so the words are always the chain's own, never a copy.
- **Follow the call**: a live trace of the visitor's own call. The browser computes the seed (keccak256 of the
  words, or of the wallet and the UTC day), calls `generate(string,uint256)`, shows the token ids one by one with
  their words from the 434-token vocabulary, and joins them exactly as the contract does.
- **Check the weights**: re-derives every part's CREATE2 address from its bytes and hashes the stream (below).

## Deployed

| | |
|---|---|
| Engine (AM3N) | [`0x2D58843F8cd565E1FbBE01Ea5d67F3d13cb91811`](https://robinhoodchain.blockscout.com/address/0x2D58843F8cd565E1FbBE01Ea5d67F3d13cb91811), source verified on [Sourcify](https://repo.sourcify.dev/4663/0x2D58843F8cd565E1FbBE01Ea5d67F3d13cb91811) (exact match) |
| Weights | 15 data contracts, listed by `parts()` and in [`docs/parts.json`](docs/parts.json) |

## Numbers

| | |
|---|---|
| Weights | 1,239,424 int8 (every expert counted), 1,247,360 bytes in 15 data contracts, keccak256 `0x51cc9ae0514c7b199cb15c97844133b9b08bb13771c44d70266fdc551884a7e2` |
| Architecture | decoder-only transformer, 2 layers, width 128, 4 heads of 32; each layer's MLP is 8 experts of width 256 with a top-1 router; tied embeddings |
| Weights a word touches | 319,744: the router runs one expert per layer, the other seven are never read |
| Vocabulary / longest blessing | 434 tokens / 10 tokens |
| Engine | 13,581 bytes |
| Gas per blessing | 29M to 37M measured on the public RPC; 40.8M for the longest possible output (10 tokens) |
| Read-call limit on the public RPC | 50M gas |

## How it works

Every operation is integer math, so the contract and the Python reference agree bit for bit.

1. **Weights on chain.** The weight stream is split into data contracts whose code is `0x00 ++ bytes` (the
   STOP byte keeps data from ever running). Each was created through the standard CREATE2 deployer
   `0x4e59b44847b379578588920cA78FbF26c0B4956C` with salt 0, so its address is derived from the hash of the
   exact bytes it holds; the engine has those addresses as constants. Embeddings, attention and routers
   (198,784 bytes) are copied into memory once per call; for each word and layer only the expert the router
   picks is copied in.
2. **Packed matrix kernel.** Weights are stored as `w + 128`, one 32-byte word per input column for a block of
   32 output rows. Shifting the word by 0, 8, 16 and 24 bits and masking one byte into each 32-bit lane gives 8
   weights; one 256-bit multiply by the input adds 8 products at once. The offset adds `128·Σx` to every row
   and is removed once. Inputs are first shifted right until every lane stays within 32 signed bits, and the
   shift is put back on the result. About 6.2 gas per multiply-accumulate.
3. **The forward pass.** Token and position embeddings, then per layer RMSNorm (integer square root),
   causal attention with a key/value cache in memory (exp from a 512-entry table), and the routed expert
   (ReLU MLP). Every rescale is an arithmetic shift; every division truncates toward zero, as Solidity does.
4. **Sampling.** The next token is drawn from the top 8 logits at temperature 0.8 with `keccak256(seed, step)`;
   ties go to the lower id. Steps 3 and 4 repeat inside the same call until the end token or 10 tokens.

## Verify

The page's "Check the weights" button does this from any browser: it asks the engine for `parts()`, reads every
part with `eth_getCode`, recomputes each CREATE2 address from the bytes it finds, and hashes the whole stream
against the keccak256 above.

```
forge test                          # 18 tests: blessings, wallets, the daily blessing, the parts, two full logit vectors
python model/live.py                # before deployment: engine + all parts injected with a state override
python model/live.py 0x2D58843F8cd565E1FbBE01Ea5d67F3d13cb91811      # after deployment: chain output vs the Python reference, with gas
```

`test/Am3n.t.sol` is generated from the reference. Its setup deploys every part through the same CREATE2
deployer and checks that each lands at the address the engine expects.

## Reproduce

```
python3.11 -m venv .venv && .venv/bin/pip install torch numpy pycryptodome
.venv/bin/python model/angel.py train   # optional: retrain (a few minutes on a laptop GPU)
.venv/bin/python model/export.py        # parts/, src/Weights.sol, test/Am3n.t.sol, docs/parts.json
forge test
```

`model/weights.json` is the trained, quantized model that is deployed: training on a GPU is not bit for bit
repeatable, so the weights themselves are committed and everything else (`parts/`, `src/Weights.sol`, the
tests) is generated from them by `export.py`; each part's CREATE2 address then proves the chain holds exactly
these bytes. The training data is synthetic: `model/corpus.py` generates
80,000 short blessings: 120 hand-written sentences and a grammar around them. No outside text is used. `deploy.py` deploys the parts and the
engine (it shows the cost and asks first).

## Limits

AM3N writes short blessings in four moods. It does not answer questions or follow instructions, and it
knows nothing beyond its corpus. The words come from a grammar, so they are simple; the point is that every
one of them is computed on chain, for free, by anyone.

## License

MIT
