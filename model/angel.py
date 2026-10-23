"""ANG3L: train a tiny transformer, quantize it to int8, and run it with integer-only math.

The integer forward pass below is the specification the Solidity contract must match bit for bit:
every operation is an integer add, multiply, arithmetic shift, truncating division, integer square
root, or table lookup.

  python model/angel.py train     # train, quantize, write model/weights.json
  python model/angel.py sample    # print blessings from the integer model
"""
import json, math, sys
from pathlib import Path
import numpy as np
from Crypto.Hash import keccak

sys.path.insert(0, str(Path(__file__).parent))
from corpus import corpus, THEMES

HERE = Path(__file__).parent
CFG = dict(d=32, L=2, H=2, F=64, T=32, A=12, K=8, MAXNEW=20)   # A: fractional bits of activations
DH = 16                                                          # head size; 1/sqrt(16) is a shift by 2
EXP_BITS = 5                                                     # exp table step is 1/32
EXP = [round(32768 * math.exp(-i / 2 ** EXP_BITS)) for i in range(512)]   # fits uint16


def vocab():
    words = sorted({w for l in corpus() for w in l.split()})
    return ["<pad>", "<eos>"] + words


# ---------------------------------------------------------------- integer model (the specification)

def sdiv(a, b):
    """Solidity's signed division: truncates toward zero. b > 0."""
    q = np.abs(a) // b
    return np.where(a < 0, -q, q)


def rmsnorm(x, g, sg, A):
    ms = int((x.astype(object) ** 2).sum()) // len(x)
    r = max(math.isqrt(ms), 1)
    return sdiv(x * g.astype(np.int64) << A, r << sg)


def mm(x, Wt, s):
    """y = x @ W, with W stored transposed (out x in) as int8, scale 2^s."""
    return (Wt.astype(np.int64) @ x) >> s


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
        Q = [mm(h, l["Wq"], l["sWq"]) for h in hs]
        K = [mm(h, l["Wk"], l["sWk"]) for h in hs]
        V = [mm(h, l["Wv"], l["sWv"]) for h in hs]
        new = []
        for t in range(len(xs)):
            heads = []
            for hd in range(H):
                sl = slice(hd * DH, (hd + 1) * DH)
                sc = np.array([(int(Q[t][sl] @ K[u][sl]) >> A) >> 2 for u in range(t + 1)], dtype=np.int64)
                e = exp_lut(sc.max() - sc, A)
                num = sum(int(e[u]) * V[u][sl] for u in range(t + 1))
                heads.append(sdiv(num, int(e.sum())))
            x = xs[t] + mm(np.concatenate(heads), l["Wo"], l["sWo"])
            h = rmsnorm(x, l["g2"], l["sg2"], A)
            new.append(x + mm(np.maximum(mm(h, l["W1"], l["sW1"]), 0), l["W2"], l["sW2"]))
        xs = new
    x = rmsnorm(xs[-1], q["gf"], q["sgf"], A)
    return mm(x, q["E"], q["sE"])                                  # tied output head


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


# ---------------------------------------------------------------- float model + training

def train():
    import torch, torch.nn as nn, torch.nn.functional as Fn
    torch.manual_seed(0)
    voc = vocab(); V, d, L, H, Fd, T = len(voc), CFG["d"], CFG["L"], CFG["H"], CFG["F"], CFG["T"]
    ids = {w: i for i, w in enumerate(voc)}
    data = torch.zeros(60000, T, dtype=torch.long)
    for n, line in enumerate(corpus()):
        seq = [ids[w] for w in line.split()] + [1]
        data[n, :len(seq)] = torch.tensor(seq)

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
                m.W1, m.W2 = p(d, Fd, sc=d ** -0.5), p(Fd, d, sc=Fd ** -0.5 * 0.5)
                s.layers.append(m)

        def forward(s, idx):
            B, n = idx.shape
            x = s.E[idx] + s.P[:n]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
            for m in s.layers:
                h = rms(x, m.g1)
                q, k, v = (h @ W for W in (m.Wq, m.Wk, m.Wv))
                q, k, v = (z.view(B, n, H, DH).transpose(1, 2) for z in (q, k, v))
                att = (q @ k.transpose(-1, -2) / 4).masked_fill(mask, -1e9).softmax(-1)
                x = x + (att @ v).transpose(1, 2).reshape(B, n, d) @ m.Wo
                x = x + torch.relu(rms(x, m.g2) @ m.W1) @ m.W2
            return rms(x, s.gf) @ s.E.T

    net = Angel(); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    tr, va = data[:57000], data[57000:]
    for step in range(3000):
        b = tr[torch.randint(0, len(tr), (256,))]
        logits = net(b[:, :-1])
        loss = Fn.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups: g["lr"] = 3e-3 * 0.5 * (1 + math.cos(math.pi * step / 3000))
        if step % 500 == 0 or step == 2999:
            with torch.no_grad():
                vl = Fn.cross_entropy(net(va[:, :-1]).reshape(-1, V), va[:, 1:].reshape(-1), ignore_index=0)
            print(f"step {step:4d}  train {loss.item():.3f}  val {vl.item():.3f}", flush=True)

    def q8(t):
        a = t.detach().numpy().astype(np.float64)
        s = int(math.floor(math.log2(127 / max(np.abs(a).max(), 1e-9))))
        return np.clip(np.round(a * 2.0 ** s), -127, 127).astype(int).tolist(), s

    out = {"cfg": CFG, "vocab": voc}
    out["E"], out["sE"] = q8(net.E); out["P"], out["sP"] = q8(net.P); out["gf"], out["sgf"] = q8(net.gf)
    out["layers"] = []
    for m in net.layers:
        l = {}
        for k in ["g1", "g2"]: l[k], l["s" + k] = q8(getattr(m, k))
        for k in ["Wq", "Wk", "Wv", "Wo", "W1", "W2"]: l[k], l["s" + k] = q8(getattr(m, k).T)   # stored out x in
        out["layers"].append(l)
    (HERE / "weights.json").write_text(json.dumps(out))
    torch.save(net.state_dict(), HERE / "float.pt")
    print("wrote model/weights.json")


def load():
    w = json.loads((HERE / "weights.json").read_text())
    arr = lambda v: np.array(v, dtype=np.int64)
    q = {k: arr(w[k]) for k in ["E", "P", "gf"]} | {k: w[k] for k in ["sE", "sP", "sgf"]}
    q["layers"] = [{k: (arr(v) if isinstance(v, list) else v) for k, v in l.items()} for l in w["layers"]]
    return q, w["vocab"]


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "sample":
        q, voc = load()
        for theme in THEMES:
            for name in ["satoshi", "0x8d6b11cfdf079e27c17e05cb7ea81235887691c6", "mom"]:
                print(f"{theme:10} {name[:12]:12} -> {bless(q, voc, theme, name)[1]}")
