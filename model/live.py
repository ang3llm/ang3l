"""Run AM3N on Robinhood Chain's public RPC and compare with the Python reference.

  python model/live.py              # before deploying: engine + all weight parts injected with a state override
  python model/live.py 0xADDRESS    # after deploying: calls the real contract

Reports the smallest gas limit each call succeeds with (the public RPC caps eth_call at 50M).
"""
import json, sys, urllib.request
from pathlib import Path
from Crypto.Hash import keccak

sys.path.insert(0, str(Path(__file__).parent))
from angel import load, bless, THEMES

RPC = "https://rpc.mainnet.chain.robinhood.com"
ROOT = Path(__file__).parent.parent
TMP = "0x00000000000000000000000000000000000a3913"
DEPLOYER = "4e59b44847b379578588920ca78fbf26c0b4956c"
k256 = lambda b: keccak.new(digest_bits=256, data=b).digest()


def rpc(method, params):
    req = urllib.request.Request(RPC, json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
                                 {"content-type": "application/json", "user-agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=180))


def calldata(words, theme):
    sel = k256(b"bless(string,uint256)")[:4]
    b = words.encode()
    pad = b + b"\0" * (-len(b) % 32)
    return "0x" + (sel + (64).to_bytes(32, "big") + theme.to_bytes(32, "big") + len(b).to_bytes(32, "big") + pad).hex()


def part_override():
    """Every weight part at its CREATE2 address, so the engine's EXTCODECOPY reads the real bytes."""
    ov = {}
    for f in sorted((ROOT / "parts").glob("*.bin")):
        init = f.read_bytes()
        runtime = init[13:]                                   # strip PUSH3 len ++ codecopy/return (13 bytes)
        addr = k256(b"\xff" + bytes.fromhex(DEPLOYER) + bytes(32) + k256(init))[12:]
        ov["0x" + addr.hex()] = {"code": "0x" + runtime.hex()}
    return ov


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
        code = json.loads((ROOT / "out/Am3n.sol/Am3n.json").read_text())["deployedBytecode"]["object"]
        override = part_override()
        override[TMP] = {"code": code if code.startswith("0x") else "0x" + code}
        print(f"state override: engine + {len(override) - 1} weight parts")
    q, voc = load()
    ok = True
    for words, theme in [("satoshi", 0), ("gm", 1), ("0x8d6b11cfdf079e27c17e05cb7ea81235887691c6", 2), ("mom", 3)]:
        data = calldata(words, theme)
        r = call(addr, data, 50_000_000, override)
        if "error" in r:
            print(f"{THEMES[theme]:10} {words[:14]:14} FAILED: {r['error']}"); ok = False; continue
        got, want = text(r["result"]), bless(q, voc, THEMES[theme], words)[1]
        lo, hi = 1_000_000, 50_000_000                        # smallest gas limit that still succeeds
        while hi - lo > 500_000:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if "result" in call(addr, data, mid, override) else (mid, hi)
        same = got == want
        ok &= same
        print(f"{THEMES[theme]:10} {words[:14]:14} {hi / 1e6:5.1f}M gas  {'MATCH' if same else 'DIFF'}  {got}")
    print("all match the Python reference" if ok else "MISMATCH")
