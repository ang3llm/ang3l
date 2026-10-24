# Contributing

AM3N is small on purpose: every word it writes is computed on chain, for free, inside one read call. Changes are
welcome when they keep that true.

- **Gas.** Most of the gas is spent in the engine's kernel and attention, in [`src/Am3n.sol`](src/Am3n.sol). A change
  that lowers the worst case (ten tokens) is welcome when every blessing stays bit for bit the same.
- **Correctness.** `python model/export.py` regenerates the parity suite from the committed weights, and
  `forge test` must pass in full. The Python reference in [`model/angel.py`](model/angel.py) is the specification:
  the contract has to agree with it on every logit and every token.
- **Research.** Experiments live in [`research/`](research/), one numbered folder each, with the question, the
  method, the measured result and how to run it.
- **Lessons.** The next model learns from the lines people write to the T3ach contract on chain; see the README.

For anything larger than a fix, open an issue first. Contributions are under the MIT license, like the rest of the
repository.
