"""Deploy AM3N to Robinhood Chain: every weight part through the CREATE2 deployer, then the engine.

  .venv/bin/python deploy.py            # reads PRIVATE_KEY from .env, shows the cost, asks before sending

Parts already on chain (anyone may deploy them; the address pins the bytes) are skipped, so a run that
stops halfway can simply be started again. Needs Foundry's `cast` and `forge`.
"""
import json, re, subprocess, sys, urllib.request
from pathlib import Path
from Crypto.Hash import keccak

ROOT = Path(__file__).parent
RPC = "https://rpc.mainnet.chain.robinhood.com"
DEPLOYER = "0x4e59b44847b379578588920ca78fbf26c0b4956c"
k256 = lambda b: keccak.new(digest_bits=256, data=b).digest()


def rpc(method, params):
    req = urllib.request.Request(RPC, json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
                                 {"content-type": "application/json", "user-agent": "Mozilla/5.0"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    if "error" in r:
        raise RuntimeError(r["error"])
    return r["result"]


def env_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("PRIVATE_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("PRIVATE_KEY missing in .env")


def run(cmd, **kw):
    """Runs a Foundry command. On failure it exits with the tool's own message, long hex runs hidden: never the
    command line, which carries the private key."""
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        sys.exit("failed: " + re.sub(r"(0x)?[0-9a-fA-F]{40,}", "<hex>", (r.stderr or r.stdout).strip())[-800:])
    return r.stdout.strip()


def cast(*args):
    return run(["cast", *args])


if __name__ == "__main__":
    key = env_key()
    me = cast("wallet", "address", "--private-key", key)
    price = int(rpc("eth_gasPrice", []), 16)
    todo = []
    for f in sorted((ROOT / "parts").glob("*.bin")):
        init = f.read_bytes()
        addr = "0x" + k256(b"\xff" + bytes.fromhex(DEPLOYER[2:]) + bytes(32) + k256(init))[12:].hex()
        if rpc("eth_getCode", [addr, "latest"]) != "0x":
            print(f"{f.name}  {addr}  already on chain")
            continue
        data = "0x" + (bytes(32) + init).hex()
        gas = int(rpc("eth_estimateGas", [{"from": me, "to": DEPLOYER, "data": data}, "latest"]), 16)
        todo.append((f.name, addr, data, gas))
    total = sum(g for *_, g in todo) + 8_000_000            # + the engine, generously
    bal = int(rpc("eth_getBalance", [me, "latest"]), 16)
    print(f"\nwallet {me}: {bal / 1e18:.6f} ETH")
    print(f"{len(todo)} parts to deploy + engine: ~{total:,} gas x {price / 1e9:.3f} gwei = ~{total * price / 1e18:.6f} ETH")
    if bal < total * price * 1.2:
        sys.exit("not enough ETH for that with a 20% margin")
    if input("send? [y/N] ").strip().lower() != "y":
        sys.exit("nothing sent")
    for name, addr, data, gas in todo:
        cast("send", DEPLOYER, data, "--private-key", key, "--rpc-url", RPC, "--gas-limit", str(gas * 12 // 10))
        assert rpc("eth_getCode", [addr, "latest"]) != "0x", f"{name} did not land at {addr}"
        print(f"{name}  {addr}  deployed")
    out = run(["forge", "create", "src/Am3n.sol:Am3n", "--broadcast", "--rpc-url", RPC, "--private-key", key], cwd=ROOT)
    engine = next(l.split()[-1] for l in out.splitlines() if l.startswith("Deployed to:"))
    print(f"\nengine: {engine}")
    print("next: put it in docs/index.html (const ADDRESS), then run")
    print(f"  forge verify-contract --verifier sourcify --chain 4663 {engine} src/Am3n.sol:Am3n")
    print(f"  .venv/bin/python model/live.py {engine}")
