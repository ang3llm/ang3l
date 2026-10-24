"""Reference for the packed dot-product kernel: 12 uint8 x uint8 products summed by one 256-bit multiply.
A = sum q_i << (20 i), B = sum u_i << (20 (11 - i)); (A * B mod 2^256) >> 220 & (2^20 - 1) == sum q_i u_i."""
import random
K, M = 20, 12
def packA(q): return sum(v << (K * i) for i, v in enumerate(q))
def packB(u): return sum(v << (K * (M - 1 - i)) for i, v in enumerate(u))
def dot(q, u): return ((packA(q) * packB(u)) % 2**256 >> (K * (M - 1))) & ((1 << K) - 1)
rng = random.Random(1)
for _ in range(200000):
    q = [rng.randint(0, 255) for _ in range(M)]; u = [rng.randint(0, 255) for _ in range(M)]
    assert dot(q, u) == sum(a * b for a, b in zip(q, u))
# worst case: every product at its maximum
assert dot([255] * M, [255] * M) == M * 255 * 255
print("packed dot product exact on 200,000 random cases and the worst case")
