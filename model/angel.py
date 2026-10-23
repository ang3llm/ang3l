"""AM3N: train a small mixture-of-experts transformer, quantize it to int8, and run it with integer-only math.

The integer forward pass below is the specification the Solidity contract must match bit for bit:
every operation is an integer add, multiply, arithmetic shift, truncating division, integer square
root, or table lookup.

  python model/angel.py train     # train, quantize, write model/weights.json
  python model/angel.py random    # random int8 weights of the real shape (for gas and parity tests)
  python model/angel.py sample    # print blessings from the integer model
"""
import json, math, sys
from pathlib import Path
import numpy as np
from Crypto.Hash import keccak

sys.path.insert(0, str(Path(__file__).parent))

HERE = Path(__file__).parent
# d model width, L layers, H heads, F expert width, NX experts per layer (one runs per token), T positions,
# A fractional bits of activations, K top-k sampling, MAXNEW longest blessing in tokens (words + end).
CFG = dict(d=128, L=2, H=4, F=256, NX=8, T=20, A=12, K=8, MAXNEW=11)
DH = 32                                                          # head size
QK = 2                                                           # attention scores are scaled by 2^-QK
EXP_BITS = 5                                                     # exp table step is 1/32
EXP = [round(32768 * math.exp(-i / 2 ** EXP_BITS)) for i in range(512)]   # fits uint16
THEMES = ["<builder>", "<holder>", "<night>", "<friend>"]


# ---------------------------------------------------------------- integer model (the specification)

def sdiv(a, b):
    """Solidity's signed division: truncates toward zero. b > 0."""
    q = np.abs(a) // b
    return np.where(a < 0, -q, q)


def rmsnorm(x, g, sg, A):
    ms = int((x.astype(object) ** 2).sum()) // len(x)
    r = max(math.isqrt(ms), 1)
    return sdiv(x * g.astype(np.int64) << A, r << sg)


def mv(Wt, x, s):
    """y = (W x) >> s, W stored transposed (out x in) as int8 with scale 2^s.

    The contract multiplies 8 weights at once inside one 256-bit word, in 32-bit lanes. To keep every lane
    in range the input is first shifted right by k until |x| <= 2^B, B = 23 - bitlen(nin - 1); the shift is
    put back on the result. For typical activations k is 0 and this is plain W x."""
    B = 23 - (len(x) - 1).bit_length()
    k = max(0, int(np.abs(x).max()).bit_length() - B)
    return ((Wt.astype(np.int64) @ (x >> k)) << k) >> s


def exp_lut(z, A):
    idx = z >> (A - EXP_BITS)
    return np.where(idx < len(EXP), np.array(EXP, dtype=np.int64)[np.minimum(idx, len(EXP) - 1)], 0)


def forward(q, tokens):
    """Integer logits for the last position. q: quantized weights (dict of int arrays + shifts)."""
    A, H = CFG["A"], CFG["H"]
    xs = [(q["E"][t].astype(np.int64) << (A - q["sE"])) + (q["P"][i].astype(np.int64) << (A - q["sP"]))
          for i, t in enumerate(tokens)]
    for l in q["layers"]:
        hs = [rmsnorm(x, l["g1"], l["sg1"], A) for x in xs]
        Q = [mv(l["Wq"], h, l["sWq"]) for h in hs]
        K = [mv(l["Wk"], h, l["sWk"]) for h in hs]
        V = [mv(l["Wv"], h, l["sWv"]) for h in hs]
        new = []
        for t in range(len(xs)):
            heads = []
            for hd in range(H):
                sl = slice(hd * DH, (hd + 1) * DH)
                sc = np.array([(int(Q[t][sl] @ K[u][sl]) >> A) >> QK for u in range(t + 1)], dtype=np.int64)
                e = exp_lut(sc.max() - sc, A)
                num = sum(int(e[u]) * V[u][sl] for u in range(t + 1))
                heads.append(sdiv(num, int(e.sum())))
            x = xs[t] + mv(l["Wo"], np.concatenate(heads), l["sWo"])
            h = rmsnorm(x, l["g2"], l["sg2"], A)
            ex = l["experts"][int(np.argmax(mv(l["Wr"], h, l["sWr"])))]   # top-1 router, ties to the lower expert
            new.append(x + mv(ex["W2"], np.maximum(mv(ex["W1"], h, ex["sW1"]), 0), ex["sW2"]))
        xs = new
    x = rmsnorm(xs[-1], q["gf"], q["sgf"], A)
    return mv(q["E"], x, q["sE"])                                  # tied output head


def sample(logits, seed, step, banned):
    """Top-K sampling at temperature 0.8, driven by keccak256(seed ++ step). Ties go to the lower id."""
    order = sorted((i for i in range(len(logits)) if i not in banned), key=lambda i: (-int(logits[i]), i))[:CFG["K"]]
    top = int(logits[order[0]])
    w = [int(exp_lut(np.int64(((top - int(logits[i])) * 5) >> 2), CFG["A"])) for i in order]
    h = keccak.new(digest_bits=256, data=seed + step.to_bytes(32, "big")).digest()
    r = int.from_bytes(h, "big") % sum(w)
    for i, wi in zip(order, w):
        if r < wi:
            return i
        r -= wi


def bless(q, voc, theme, text):
    seed = keccak.new(digest_bits=256, data=text.encode()).digest()
    banned = {voc.index(w) for w in ["<pad>"] + THEMES}
    toks = [voc.index(theme)]
    for step in range(CFG["MAXNEW"]):
        t = sample(forward(q, toks), seed, step, banned)
        if t == 1:
            break
        toks.append(t)
    return toks, detok([voc[t] for t in toks[1:]])


def detok(words):
    out = ""
    for w in words:
        out += w if (w in ",.;" or not out) else " " + w
    return out


# ---------------------------------------------------------------- weights file

def save(out):
    (HERE / "weights.json").write_text(json.dumps(out))


def load():
    w = json.loads((HERE / "weights.json").read_text())
    arr = lambda v: np.array(v, dtype=np.int64)
    q = {k: arr(w[k]) for k in ["E", "P", "gf"]} | {k: w[k] for k in ["sE", "sP", "sgf"]}
    q["layers"] = []
    for l in w["layers"]:
        d = {k: (arr(v) if isinstance(v, list) else v) for k, v in l.items() if k != "experts"}
        d["experts"] = [{k: (arr(v) if isinstance(v, list) else v) for k, v in e.items()} for e in l["experts"]]
        q["layers"].append(d)
    return q, w["vocab"]


def random_weights(V=640, seed=1):
    """Random int8 weights of the real shape, with shifts that keep activations near unit scale.
    The end token's embedding is zero, so its logit stays mid-pack and generation runs to MAXNEW (worst case)."""
    rng = np.random.default_rng(seed)
    d, L, F, NX, T = (CFG[k] for k in ("d", "L", "F", "NX", "T"))
    m = lambda *sh: rng.integers(-127, 128, size=sh).tolist()
    gain = lambda: rng.integers(48, 80, size=d).tolist()          # ~1.0 at scale 2^6
    E = rng.integers(-127, 128, size=(V, d)); E[1] = 0
    out = {"cfg": CFG, "vocab": ["<pad>", "<eos>"] + THEMES + [f"w{i}" for i in range(V - 6)],
           "E": E.tolist(), "sE": 10, "P": m(T, d), "sP": 10, "gf": gain(), "sgf": 6, "layers": []}
    for _ in range(L):
        l = {"g1": gain(), "sg1": 6, "g2": gain(), "sg2": 6, "Wr": m(NX, d), "sWr": 10}
        for k in ["Wq", "Wk", "Wv", "Wo"]:
            l[k], l["s" + k] = m(d, d), 10
        l["experts"] = [{"W1": m(F, d), "sW1": 10, "W2": m(d, F), "sW2": 11} for _ in range(NX)]
        out["layers"].append(l)
    return out


# ---------------------------------------------------------------- float model + training

def train(steps=4000):
    import torch, torch.nn as nn, torch.nn.functional as Fn
    from corpus import corpus
    torch.manual_seed(0)
    lines = corpus()
    voc = ["<pad>", "<eos>"] + sorted({w for l in lines for w in l.split()})
    V, d, L, H, Fd, NX, T = len(voc), CFG["d"], CFG["L"], CFG["H"], CFG["F"], CFG["NX"], CFG["T"]
    ids = {w: i for i, w in enumerate(voc)}
    data = torch.zeros(len(lines), T + 1, dtype=torch.long)
    for n, line in enumerate(lines):
        seq = [ids[w] for w in line.split()] + [1]
        data[n, :len(seq)] = torch.tensor(seq)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    def rms(x, g): return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * g

    class Angel(nn.Module):
        def __init__(s):
            super().__init__()
            p = lambda *sh, sc: nn.Parameter(torch.randn(*sh) * sc)
            s.E, s.P, s.gf = p(V, d, sc=0.1), p(T, d, sc=0.05), nn.Parameter(torch.ones(d))
            s.layers = nn.ModuleList()
            for _ in range(L):
                m = nn.Module()
                m.g1, m.g2 = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d))
                for k in ["Wq", "Wk", "Wv", "Wo"]: setattr(m, k, p(d, d, sc=d ** -0.5))
                m.Wr = p(d, NX, sc=0.01)
                m.W1, m.W2 = p(NX, d, Fd, sc=d ** -0.5), p(NX, Fd, d, sc=Fd ** -0.5 * 0.5)
                s.layers.append(m)

        def forward(s, idx):
            B, n = idx.shape
            x = s.E[idx] + s.P[:n]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=idx.device), 1)
            aux = 0
            for m in s.layers:
                h = rms(x, m.g1)
                q, k, v = (h @ W for W in (m.Wq, m.Wk, m.Wv))
                q, k, v = (z.view(B, n, H, DH).transpose(1, 2) for z in (q, k, v))
                att = (q @ k.transpose(-1, -2) / 2 ** QK).masked_fill(mask, -1e9).softmax(-1)
                x = x + (att @ v).transpose(1, 2).reshape(B, n, d) @ m.Wo
                h = rms(x, m.g2)
                pr = (h @ m.Wr).softmax(-1)                                  # B n NX
                e = pr.argmax(-1)
                ys = torch.einsum("bnf,xfd->bnxd", torch.relu(torch.einsum("bnd,xdf->bnxf", h, m.W1)), m.W2)
                y = ys.gather(2, e[..., None, None].expand(B, n, 1, d)).squeeze(2)
                g = pr.gather(-1, e[..., None])
                x = x + y * (g / g.detach())          # forward value is exactly the expert's output; the router still learns
                frac = Fn.one_hot(e, NX).float().mean((0, 1))
                aux = aux + NX * (frac * pr.mean((0, 1))).sum()
            return rms(x, s.gf) @ s.E.T, aux

    net = Angel().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.01)
    split = len(data) * 95 // 100
    tr, va = data[:split].to(dev), data[split:].to(dev)
    for step in range(steps):
        b = tr[torch.randint(0, len(tr), (256,), device=dev)]
        logits, aux = net(b[:, :-1])
        loss = Fn.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1), ignore_index=0)
        opt.zero_grad(); (loss + 0.01 * aux).backward(); opt.step()
        for g in opt.param_groups: g["lr"] = 2e-3 * 0.5 * (1 + math.cos(math.pi * step / steps))
        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                vl = Fn.cross_entropy(net(va[:2000, :-1])[0].reshape(-1, V), va[:2000, 1:].reshape(-1), ignore_index=0)
            print(f"step {step:5d}  train {loss.item():.3f}  val {vl.item():.3f}  aux {aux.item():.3f}", flush=True)

    def q8(t):
        a = t.detach().cpu().numpy().astype(np.float64)
        s = int(math.floor(math.log2(127 / max(np.abs(a).max(), 1e-9))))
        return np.clip(np.round(a * 2.0 ** s), -127, 127).astype(int).tolist(), s

    out = {"cfg": CFG, "vocab": voc}
    out["E"], out["sE"] = q8(net.E); out["P"], out["sP"] = q8(net.P); out["gf"], out["sgf"] = q8(net.gf)
    out["layers"] = []
    for m in net.layers:
        l = {}
        for k in ["g1", "g2"]: l[k], l["s" + k] = q8(getattr(m, k))
        for k in ["Wq", "Wk", "Wv", "Wo", "Wr"]: l[k], l["s" + k] = q8(getattr(m, k).T)   # stored out x in
        l["experts"] = []
        for x in range(NX):
            e = {}
            e["W1"], e["sW1"] = q8(m.W1[x].T); e["W2"], e["sW2"] = q8(m.W2[x].T)
            l["experts"].append(e)
        out["layers"].append(l)
    save(out)
    torch.save(net.state_dict(), HERE / "float.pt")
    print("wrote model/weights.json")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "random":
        save(random_weights()); print("wrote random model/weights.json")
    elif sys.argv[1] == "sample":
        q, voc = load()
        for theme in THEMES:
            for name in ["satoshi", "0x8d6b11cfdf079e27c17e05cb7ea81235887691c6", "mom"]:
                print(f"{theme:10} {name[:12]:12} -> {bless(q, voc, theme, name)[1]}")
