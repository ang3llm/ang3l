// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

import "./Weights.sol";

/// @title AM3N
/// @notice A mixture-of-experts transformer that lives on Robinhood Chain. Its weights sit in data contracts
/// whose CREATE2 addresses are derived from the exact bytes they hold; this contract runs the whole model
/// and writes a blessing inside one read-only call. Every operation is integer math, so any node returns
/// the same words for the same input.
contract Am3n {
    error UnknownTheme();

    /// Working memory for one call.
    struct Ctx {
        bytes res; // resident weights: embeddings, positions, attention, routers, norms
        bytes xw; // the expert in use: W1 then W2
        int256[] kc; // keys, [layer][position][D]
        int256[] vc; // values
        bytes ex; // exp table
    }

    /// @notice Write a blessing. `words` seeds the sampler and is the name the angel uses (a name, an address,
    /// anything); `theme` is 0 builder, 1 holder, 2 night, 3 friend. Same input, same blessing.
    function bless(string calldata words, uint256 theme) external view returns (string memory) {
        return string(_text(_generate(keccak256(bytes(words)), theme), _name(bytes(words))));
    }

    /// @notice Bless a wallet. The angel reads it in the same call: a contract gets the builder's blessing,
    /// 0.1 ETH or more the holder's, an empty wallet the night's, anything else the friend's. New words each day.
    function blessAddress(address who) external view returns (string memory text, uint256 theme) {
        theme = who.code.length > 0 ? 0 : who.balance >= 0.1 ether ? 1 : who.balance == 0 ? 2 : 3;
        bytes32 seed = keccak256(abi.encodePacked(who, block.timestamp / 1 days));
        text = string(_text(_generate(seed, theme), _short(who)));
    }

    /// @notice Today's blessing, the same for everyone until 00:00 UTC.
    function today() external view returns (string memory) {
        uint256 day = block.timestamp / 1 days;
        return string(_text(_generate(keccak256(abi.encodePacked("am3n", day)), day % 4), "friend"));
    }

    /// @notice The token ids behind `bless`, theme token first.
    function generate(string calldata words, uint256 theme) external view returns (uint256[] memory) {
        return _generate(keccak256(bytes(words)), theme);
    }

    /// @notice Raw integer logits for the token after `tokens` (forward passes only, no sampling).
    function logits(uint256[] calldata tokens) external view returns (int256[] memory lg) {
        require(tokens.length > 0 && tokens.length <= MAXNEW, "length");
        Ctx memory c = _ctx();
        int256[] memory x;
        for (uint256 p; p < tokens.length; ++p) {
            require(tokens[p] < V, "token");
            x = _position(c, tokens[p], p);
        }
        lg = _head(c, x);
    }

    /// @notice Number of weights in the model, every expert counted.
    function parameters() external pure returns (uint256) {
        return PARAMS;
    }

    function _ctx() private view returns (Ctx memory c) {
        c.res = new bytes(RES);
        _copy(c.res, 0, RES);
        c.xw = new bytes(XSZ);
        c.kc = new int256[](L * MAXNEW * D);
        c.vc = new int256[](L * MAXNEW * D);
        c.ex = EXPS;
    }

    function _generate(bytes32 seed, uint256 theme) private view returns (uint256[] memory out) {
        if (theme >= 4) revert UnknownTheme();
        Ctx memory c = _ctx();
        out = new uint256[](MAXNEW + 1);
        uint256 tok = [THEME0, THEME1, THEME2, THEME3][theme];
        out[0] = tok;
        uint256 n = 1;
        for (uint256 step; step < MAXNEW; ++step) {
            uint256 mark;
            assembly { mark := mload(0x40) }
            tok = _sample(_head(c, _position(c, tok, step)), c.ex, seed, step);
            assembly { mstore(0x40, mark) } // this step's arrays are dead: hand their memory to the next step
            if (tok == EOS) break;
            out[n++] = tok;
        }
        assembly { mstore(out, n) } // trim to the tokens actually written
    }

    /// Hidden state of the token at position `p`; fills this position's keys and values.
    function _position(Ctx memory c, uint256 tok, uint256 p) private view returns (int256[] memory x) {
        x = new int256[](D);
        bytes memory w = c.res;
        uint256 r = tok % 32;
        uint256 at = (tok / 32) * D * 32 + 31 - 4 * (r % 8) - r / 8; // row `tok` of packed E, input column 0
        unchecked {
            for (uint256 i; i < D; ++i) {
                x[i] = (int256(uint256(uint8(w[at + 32 * i]))) - 128) * int256(1 << (A - SE))
                    + int256(int8(uint8(w[OFF_P + p * D + i]))) * int256(1 << (A - SP));
            }
        }
        for (uint256 l; l < L; ++l) {
            _attnBlock(c, x, l, p);
            _moeBlock(c, x, l);
        }
    }

    /// x += Wo · attention(RMSNorm(x)); this position's keys and values go into the caches.
    function _attnBlock(Ctx memory c, int256[] memory x, uint256 l, uint256 p) private pure {
        uint256 o = OFF_L + l * LAYER;
        bytes memory w = c.res;
        int256[] memory h = _norm(w, o, _shift(l, 0), x);
        uint256 at = (l * MAXNEW + p) * D;
        _store(c.kc, at, _mv(w, o + D + D * D, h, D, _shift(l, 2)));
        _store(c.vc, at, _mv(w, o + D + 2 * D * D, h, D, _shift(l, 3)));
        int256[] memory att = _attend(_mv(w, o + D, h, D, _shift(l, 1)), c, l * MAXNEW * D, p);
        _addTo(x, _mv(w, o + D + 3 * D * D, att, D, _shift(l, 4)));
    }

    /// x += W2 · relu(W1 · RMSNorm(x)) of the one expert the router picks; only that expert is read.
    function _moeBlock(Ctx memory c, int256[] memory x, uint256 l) private view {
        uint256 o = OFF_L + l * LAYER;
        int256[] memory h = _norm(c.res, o + D + 4 * D * D, _shift(l, 5), x);
        int256[] memory r = _mv(c.res, o + 2 * D + 4 * D * D, h, 32, _shift(l, 6));
        uint256 e;
        for (uint256 i = 1; i < NX; ++i) {
            if (r[i] > r[e]) e = i;
        }
        _copy(c.xw, RES + (l * NX + e) * XSZ, XSZ);
        int256[] memory f = _mv(c.xw, 0, h, F, _shift(l, 7 + 2 * e));
        unchecked {
            for (uint256 i; i < F; ++i) {
                if (f[i] < 0) f[i] = 0;
            }
        }
        _addTo(x, _mv(c.xw, F * D, f, D, _shift(l, 8 + 2 * e)));
    }

    function _head(Ctx memory c, int256[] memory x) private pure returns (int256[] memory lg) {
        lg = _mv(c.res, 0, _norm(c.res, OFF_GF, SGF, x), VP, SE); // tied output head
        assembly { mstore(lg, V) } // drop the padding rows
    }

    /// Causal attention for the newest position `p`, all heads; this layer's keys and values start at `base`.
    function _attend(int256[] memory q, Ctx memory c, uint256 base, uint256 p) private pure returns (int256[] memory out) {
        out = new int256[](D);
        int256[] memory s = new int256[](p + 1);
        for (uint256 c0; c0 < D * 32; c0 += DH * 32) {
            int256 top = _scores(q, c.kc, base, c0, s);
            _mix(s, c.vc, base, c0, _weigh(s, top, c.ex), out);
        }
    }

    /// s[u] = (q · k_u >> A) >> QK over one head (byte offset c0 in a row); returns the largest.
    function _scores(int256[] memory q, int256[] memory kc, uint256 base, uint256 c0, int256[] memory s)
        private pure returns (int256 top)
    {
        assembly {
            top := shl(255, 1)
            let qp := add(add(q, 32), c0)
            let kp := add(add(add(kc, 32), shl(5, base)), c0)
            let sp := add(s, 32)
            let se := add(sp, shl(5, mload(s)))
            for {} lt(sp, se) { sp := add(sp, 32) } {
                let dot := 0
                for { let i := 0 } lt(i, shl(5, DH)) { i := add(i, 128) } {
                    dot := add(dot, mul(mload(add(qp, i)), mload(add(kp, i))))
                    dot := add(dot, mul(mload(add(qp, add(i, 32))), mload(add(kp, add(i, 32)))))
                    dot := add(dot, mul(mload(add(qp, add(i, 64))), mload(add(kp, add(i, 64)))))
                    dot := add(dot, mul(mload(add(qp, add(i, 96))), mload(add(kp, add(i, 96)))))
                }
                let sc := sar(QK, sar(A, dot))
                mstore(sp, sc)
                if sgt(sc, top) { top := sc }
                kp := add(kp, shl(5, D)) // next position
            }
        }
    }

    /// s[u] = exp(s[u] - top) from the table; returns the sum.
    function _weigh(int256[] memory s, int256 top, bytes memory ex) private pure returns (int256 den) {
        assembly {
            let ep := add(ex, 32)
            let sp := add(s, 32)
            let se := add(sp, shl(5, mload(s)))
            for {} lt(sp, se) { sp := add(sp, 32) } {
                let i := shr(sub(A, 5), sub(top, mload(sp))) // table step 1/32
                let e := 0
                if lt(i, 512) { e := shr(240, mload(add(ep, shl(1, i)))) }
                mstore(sp, e)
                den := add(den, e)
            }
        }
    }

    /// out[head] = Σ_u s[u] v_u / den, truncating toward zero.
    function _mix(int256[] memory s, int256[] memory vc, uint256 base, uint256 c0, int256 den, int256[] memory out)
        private pure
    {
        assembly {
            let n := mload(s)
            let sp := add(s, 32)
            let vp := add(add(add(vc, 32), shl(5, base)), c0)
            let op := add(add(out, 32), c0)
            for { let i := 0 } lt(i, shl(5, DH)) { i := add(i, 32) } {
                let num := 0
                let at := add(vp, i)
                for { let u := 0 } lt(u, n) { u := add(u, 1) } {
                    num := add(num, mul(mload(add(sp, shl(5, u))), mload(at)))
                    at := add(at, shl(5, D))
                }
                mstore(add(op, i), sdiv(num, den))
            }
        }
    }

    /// y = (W x) >> sh for W of `nout` rows (a multiple of 32) at byte `off` of `w`.
    ///
    /// Layout: weights stored as w + 128; one 32-byte word holds input column i for a block of 32 output rows,
    /// row 8p + L at byte 31 - 4L - p. Shifting by 8p and masking one byte per 32-bit lane gives 8 weights;
    /// one multiply by x[i] adds 8 products. The offset adds 128·Σx to every row, removed once at the end.
    /// x is first shifted right by k so that no lane leaves 32 signed bits (mv in model/angel.py).
    function _mv(bytes memory w, uint256 off, int256[] memory x, uint256 nout, uint256 sh)
        private pure returns (int256[] memory y)
    {
        y = new int256[](nout);
        (uint256 k, int256 corr, uint256 s0, uint256 se) = _shiftIn(x);
        uint256 wp;
        uint256 yp;
        assembly {
            wp := add(add(w, 32), off)
            yp := add(y, 32)
        }
        _kernel(sh, corr, k, yp + nout * 32, yp, se, s0, wp);
    }

    /// Writes x >> k to scratch memory past the free pointer, k = max(0, bitlen(max|x|) + bitlen(n - 1) - 23).
    function _shiftIn(int256[] memory x) private pure returns (uint256 k, int256 corr, uint256 s0, uint256 se) {
        assembly {
            let n := mload(x)
            let xp := add(x, 32)
            let xe := add(xp, shl(5, n))
            let m := 0
            for { let p := xp } lt(p, xe) { p := add(p, 32) } {
                let v := mload(p)
                if slt(v, 0) { v := sub(0, v) }
                if gt(v, m) { m := v }
            }
            let b := 0
            for {} m { m := shr(1, m) } { b := add(b, 1) }
            for { let t := sub(n, 1) } t { t := shr(1, t) } { b := add(b, 1) }
            if gt(b, 23) { k := sub(b, 23) }
            s0 := mload(0x40) // not allocated: nothing else uses memory until the kernel is done
            se := add(s0, shl(5, n))
            let sx := 0
            for { let q := s0 } lt(q, se) { q := add(q, 32) } {
                let v := sar(k, mload(xp))
                mstore(q, v)
                sx := add(sx, v)
                xp := add(xp, 32)
            }
            corr := shl(7, sx)
        }
    }

    function _kernel(uint256 sh, int256 corr, uint256 k, uint256 ye, uint256 yp, uint256 se, uint256 s0, uint256 wp)
        private pure
    {
        assembly {
            let M := 0x000000ff000000ff000000ff000000ff000000ff000000ff000000ff000000ff
            for {} lt(yp, ye) { yp := add(yp, 1024) } {
                let a0 := 0
                let a1 := 0
                let a2 := 0
                let a3 := 0
                let W := 0
                let v := 0
                for { let p := s0 } lt(p, se) { p := add(p, 128) } {
                        W := mload(wp)
                        v := mload(p)
                        a0 := add(a0, mul(and(W, M), v))
                        a1 := add(a1, mul(and(shr(8, W), M), v))
                        a2 := add(a2, mul(and(shr(16, W), M), v))
                        a3 := add(a3, mul(and(shr(24, W), M), v))
                        W := mload(add(wp, 32))
                        v := mload(add(p, 32))
                        a0 := add(a0, mul(and(W, M), v))
                        a1 := add(a1, mul(and(shr(8, W), M), v))
                        a2 := add(a2, mul(and(shr(16, W), M), v))
                        a3 := add(a3, mul(and(shr(24, W), M), v))
                        W := mload(add(wp, 64))
                        v := mload(add(p, 64))
                        a0 := add(a0, mul(and(W, M), v))
                        a1 := add(a1, mul(and(shr(8, W), M), v))
                        a2 := add(a2, mul(and(shr(16, W), M), v))
                        a3 := add(a3, mul(and(shr(24, W), M), v))
                        W := mload(add(wp, 96))
                        v := mload(add(p, 96))
                        a0 := add(a0, mul(and(W, M), v))
                        a1 := add(a1, mul(and(shr(8, W), M), v))
                        a2 := add(a2, mul(and(shr(16, W), M), v))
                        a3 := add(a3, mul(and(shr(24, W), M), v))
                    wp := add(wp, 128)
                }
                // each accumulator holds 8 signed 32-bit lanes: peel the lowest, remove it, shift down
                for { let o := 0 } lt(o, 256) { o := add(o, 32) } {
                    let z := signextend(3, a0)
                    mstore(add(yp, o), sar(sh, shl(k, sub(z, corr))))
                    a0 := sar(32, sub(a0, z))
                    z := signextend(3, a1)
                    mstore(add(yp, add(o, 256)), sar(sh, shl(k, sub(z, corr))))
                    a1 := sar(32, sub(a1, z))
                    z := signextend(3, a2)
                    mstore(add(yp, add(o, 512)), sar(sh, shl(k, sub(z, corr))))
                    a2 := sar(32, sub(a2, z))
                    z := signextend(3, a3)
                    mstore(add(yp, add(o, 768)), sar(sh, shl(k, sub(z, corr))))
                    a3 := sar(32, sub(a3, z))
                }
            }
        }
    }

    /// Copies `len` bytes of the weight stream from byte `off` into `dst`, across part boundaries.
    function _copy(bytes memory dst, uint256 off, uint256 len) private view {
        bytes memory parts = PARTS;
        assembly {
            let d := add(dst, 32)
            for {} len {} {
                let o := mod(off, CHUNK)
                let n := sub(CHUNK, o)
                if gt(n, len) { n := len }
                extcodecopy(shr(96, mload(add(add(parts, 32), mul(div(off, CHUNK), 20)))), d, add(o, 1), n)
                d := add(d, n)
                off := add(off, n)
                len := sub(len, n)
            }
        }
    }

    /// Top-K sampling at temperature 0.8; randomness is keccak256(seed, step). Ties go to the lower id.
    function _sample(int256[] memory lg, bytes memory ex, bytes32 seed, uint256 step) private pure returns (uint256) {
        uint256[K] memory id;
        int256[K] memory v;
        uint256 n;
        unchecked {
            for (uint256 t; t < V; ++t) {
                if (t == PAD || t == THEME0 || t == THEME1 || t == THEME2 || t == THEME3) continue;
                if (n == K && lg[t] <= v[K - 1]) continue;
                uint256 j = n < K ? n++ : K - 1;
                while (j > 0 && v[j - 1] < lg[t]) {
                    v[j] = v[j - 1];
                    id[j] = id[j - 1];
                    --j;
                }
                v[j] = lg[t];
                id[j] = t;
            }
            uint256[K] memory e;
            uint256 sum;
            for (uint256 k; k < K; ++k) {
                e[k] = _exp(ex, uint256(((v[0] - v[k]) * 5) >> 2));
                sum += e[k];
            }
            uint256 r = uint256(keccak256(abi.encodePacked(seed, step))) % sum;
            for (uint256 k; k < K; ++k) {
                if (r < e[k]) return id[k];
                r -= e[k];
            }
        }
        return id[K - 1];
    }

    /// exp(-z / 2^A) in units of 2^15, from a 512-entry table with step 1/32.
    function _exp(bytes memory ex, uint256 z) private pure returns (uint256 e) {
        uint256 i = z >> (A - 5);
        if (i >= 512) return 0;
        assembly { e := shr(240, mload(add(add(ex, 32), shl(1, i)))) }
    }

    /// RMSNorm: x * g / rms(x), integer square root, division truncates toward zero.
    function _norm(bytes memory w, uint256 off, uint256 sg, int256[] memory x) private pure returns (int256[] memory y) {
        uint256 n = x.length;
        uint256 ss;
        unchecked {
            for (uint256 i; i < n; ++i) ss += uint256(x[i] * x[i]);
        }
        uint256 r = _sqrt(ss / n);
        if (r == 0) r = 1;
        int256 den = int256(r << sg);
        y = new int256[](n);
        unchecked {
            for (uint256 i; i < n; ++i) y[i] = (x[i] * int256(int8(uint8(w[off + i]))) * int256(1 << A)) / den;
        }
    }

    function _sqrt(uint256 x) private pure returns (uint256 z) {
        if (x == 0) return 0;
        z = x;
        uint256 y = (x + 1) / 2;
        while (y < z) {
            z = y;
            y = (x / y + y) / 2;
        }
    }

    /// Detokenize: words joined by spaces, punctuation glued to the word before, <name> replaced by `name`.
    function _text(uint256[] memory toks, bytes memory name) private pure returns (bytes memory out) {
        bytes memory voc = VOCAB;
        bytes memory off = VOFF;
        for (uint256 k = 1; k < toks.length; ++k) {
            if (toks[k] == NAME) {
                out = out.length > 0 ? bytes.concat(out, " ", name) : name;
                continue;
            }
            uint256 a = _u16(off, toks[k]);
            uint256 b = _u16(off, toks[k] + 1);
            bytes1 c = voc[a];
            bool glue = b - a == 1 && (c == "," || c == "." || c == ";");
            if (out.length > 0 && !glue) out = bytes.concat(out, " ");
            bytes memory word = new bytes(b - a);
            for (uint256 i; i < b - a; ++i) word[i] = voc[a + i];
            out = bytes.concat(out, word);
        }
    }

    /// The name as written: a 42-character 0x address is shortened to 0x1234…abcd; anything else keeps
    /// printable ASCII only, at most 24 bytes; nothing left means "friend".
    function _name(bytes calldata b) private pure returns (bytes memory out) {
        if (b.length == 42 && b[0] == "0" && b[1] == "x") return bytes.concat(b[:6], unicode"…", b[38:]);
        out = new bytes(24);
        uint256 n;
        for (uint256 i; i < b.length && n < 24; ++i) {
            if (b[i] >= 0x20 && b[i] <= 0x7e) out[n++] = b[i];
        }
        if (n == 0) return "friend";
        assembly { mstore(out, n) }
    }

    function _short(address who) private pure returns (bytes memory) {
        bytes20 a = bytes20(who);
        bytes memory hexd = "0123456789abcdef";
        bytes memory s = new bytes(4);
        bytes memory e = new bytes(4);
        for (uint256 i; i < 2; ++i) {
            s[2 * i] = hexd[uint8(a[i]) >> 4];
            s[2 * i + 1] = hexd[uint8(a[i]) & 15];
            e[2 * i] = hexd[uint8(a[18 + i]) >> 4];
            e[2 * i + 1] = hexd[uint8(a[18 + i]) & 15];
        }
        return bytes.concat("0x", s, unicode"…", e);
    }

    function _u16(bytes memory b, uint256 i) private pure returns (uint256) {
        return (uint256(uint8(b[2 * i])) << 8) | uint8(b[2 * i + 1]);
    }

    /// Shift k of layer `l`: 0 g1, 1 Wq, 2 Wk, 3 Wv, 4 Wo, 5 g2, 6 Wr, then 7 + 2e W1 and 8 + 2e W2 of expert e.
    function _shift(uint256 l, uint256 k) private pure returns (uint256) {
        return uint8(SHIFTS[l * (7 + 2 * NX) + k]);
    }

    function _store(int256[] memory cache, uint256 at, int256[] memory v) private pure {
        unchecked {
            for (uint256 i; i < v.length; ++i) cache[at + i] = v[i];
        }
    }

    function _addTo(int256[] memory x, int256[] memory y) private pure {
        unchecked {
            for (uint256 i; i < x.length; ++i) x[i] += y[i];
        }
    }
}
