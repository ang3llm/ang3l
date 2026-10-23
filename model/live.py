"""Run ANG3L on Robinhood Chain's public RPC and compare with the Python reference.

  python model/live.py              # before deploying: the compiled code is injected with a state override
  python model/live.py 0xADDRESS    # after deploying: calls the real contract

Reports the smallest gas limit each call succeeds with (the public RPC caps eth_call at 50M).
"""
import json, sys, urllib.request
from pathlib import Path
from Crypto.Hash import keccak

sys.path.insert(0, str(Path(__file__).parent))
from angel import load, bless
from corpus import THEMES

RPC = "https://rpc.mainnet.chain.robinhood.com"
ROOT = Path(__file__).parent.parent
TMP = "0x00000000000000000000000000000000000a3913"


def rpc(method, params):
    req = urllib.request.Request(RPC, json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
                                 {"content-type": "application/json", "user-agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def calldata(words, theme):
    sel = keccak.new(digest_bits=256, data=b"bless(string,uint256)").digest()[:4]
    b = words.encode()
    pad = b + b"\0" * (-len(b) % 32)
    return "0x" + (sel + (64).to_bytes(32, "big") + theme.to_bytes(32, "big") + len(b).to_bytes(32, "big") + pad).hex()


def call(to, data, gas, override):
    params = [{"to": to, "data": data, "gas": hex(gas)}, "latest"] + ([override] if override else [])
    return rpc("eth_call", params)


def text(res):
    raw = bytes.fromhex(res[2:])
    n = int.from_bytes(raw[32:64], "big")
    return raw[64:64 + n].decode()


if __name__ == "__main__":
    addr = sys.argv[1] if len(sys.argv) > 1 else TMP
    override = None
    if len(sys.argv) == 1:
        code = json.loads((ROOT / "out/Ang3l.sol/Ang3l.json").read_text())["deployedBytecode"]["object"]
        override = {TMP: {"code": code}}
    q, voc = load()
    ok = True
    for words, theme in [("satoshi", 0), ("gm", 1), ("0x8d6b11cfdf079e27c17e05cb7ea81235887691c6", 2), ("mom", 3)]:
        data = calldata(words, theme)
        r = call(addr, data, 50_000_000, override)
        if "error" in r:
            print(f"{THEMES[theme]:10} {words[:14]:14} FAILED: {r['error']}"); ok = False; continue
        got, want = text(r["result"]), bless(q, voc, THEMES[theme], words)[1]
        lo, hi = 1_000_000, 50_000_000          # smallest gas limit that still succeeds
        while hi - lo > 250_000:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if "result" in call(addr, data, mid, override) else (mid, hi)
        same = got == want
        ok &= same
        print(f"{THEMES[theme]:10} {words[:14]:14} {hi / 1e6:5.1f}M gas  {'MATCH' if same else 'DIFF'}  {got}")
    print("all match the Python reference" if ok else "MISMATCH")
