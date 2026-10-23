// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

import "./Weights.sol";

/// @title ANG3L
/// @notice A transformer language model that lives in this contract. Weights, tokenizer and the
/// forward pass are all here; a whole blessing is written inside one read-only call.
/// Every operation is integer math, so any node returns the same words for the same input.
contract Ang3l {
    error UnknownTheme();

    /// @notice Write a blessing. `words` seeds the sampler (a name, an address, anything);
    /// `theme` is 0 builder, 1 holder, 2 night, 3 friend. Same input, same blessing.
    function bless(string calldata words, uint256 theme) external pure returns (string memory) {
        return string(_text(_generate(keccak256(bytes(words)), theme)));
    }

    /// @notice The token ids behind `bless`, theme token first.
    function generate(string calldata words, uint256 theme) external pure returns (uint256[] memory) {
        return _generate(keccak256(bytes(words)), theme);
    }

    /// @notice Raw integer logits for the token after `tokens` (one forward pass, no sampling).
    function logits(uint256[] calldata tokens) external pure returns (int256[] memory) {
        require(tokens.length > 0 && tokens.length <= T, "length");
        bytes memory w = _weights();
        bytes memory ex = EXPS;
        int256[] memory kc = new int256[](L * T * D);
        int256[] memory vc = new int256[](L * T * D);
        int256[] memory x;
        for (uint256 p; p < tokens.length; ++p) {
            require(tokens[p] < V, "token");
            x = _position(w, ex, kc, vc, tokens[p], p);
        }
        return _mv(w, 0, _norm(w, OFF_GF, SGF, x), V, SE);
    }

    /// @notice Number of int8 weights in the model.
    function parameters() external pure returns (uint256) {
        return _weights().length;
    }

    function _weights() private pure returns (bytes memory) {
        return WEIGHTS; // the only reference, so the weights are stored once in the bytecode
    }

    function _generate(bytes32 seed, uint256 theme) private pure returns (uint256[] memory out) {
        if (theme >= 4) revert UnknownTheme();
        bytes memory w = _weights();
        bytes memory ex = EXPS;
        int256[] memory kc = new int256[](L * T * D);
        int256[] memory vc = new int256[](L * T * D);
        out = new uint256[](MAXNEW + 1);
        uint256 tok = [THEME0, THEME1, THEME2, THEME3][theme];
        out[0] = tok;
        uint256 n = 1;
        for (uint256 step; step < MAXNEW; ++step) {
            int256[] memory x = _position(w, ex, kc, vc, tok, step);
            tok = _sample(_mv(w, 0, _norm(w, OFF_GF, SGF, x), V, SE), ex, seed, step);
            if (tok == EOS) break;
            out[n++] = tok;
        }
        assembly { mstore(out, n) } // trim to the tokens actually written
    }

    /// Hidden state of the token at position `p`; fills this position's keys and values.
    function _position(bytes memory w, bytes memory ex, int256[] memory kc, int256[] memory vc, uint256 tok, uint256 p)
        private pure returns (int256[] memory x)
    {
        x = new int256[](D);
        unchecked {
            for (uint256 i; i < D; ++i) {
                x[i] = int256(int8(uint8(w[tok * D + i]))) * int256(1 << (A - SE))
                    + int256(int8(uint8(w[OFF_P + p * D + i]))) * int256(1 << (A - SP));
            }
        }
        for (uint256 l; l < L; ++l) {
            _attnBlock(w, ex, kc, vc, x, l, p);
            _mlpBlock(w, x, l);
        }
    }

    /// x += Wo · attention(RMSNorm(x)); this position's keys and values go into the caches.
    function _attnBlock(bytes memory w, bytes memory ex, int256[] memory kc, int256[] memory vc, int256[] memory x, uint256 l, uint256 p)
        private pure
    {
        uint256 o = OFF_L + l * LAYER;
        int256[] memory h = _norm(w, o, _shift(l, 0), x);
        _store(kc, (l * T + p) * D, _mv(w, o + D + D * D, h, D, _shift(l, 2)));
        _store(vc, (l * T + p) * D, _mv(w, o + D + 2 * D * D, h, D, _shift(l, 3)));
        int256[] memory att = _attend(_mv(w, o + D, h, D, _shift(l, 1)), ex, kc, vc, l * T * D, p);
        _addTo(x, _mv(w, o + D + 3 * D * D, att, D, _shift(l, 4)));
    }

    /// x += W2 · relu(W1 · RMSNorm(x)).
    function _mlpBlock(bytes memory w, int256[] memory x, uint256 l) private pure {
        uint256 o = OFF_L + l * LAYER;
        int256[] memory f = _mv(w, o + 2 * D + 4 * D * D, _norm(w, o + D + 4 * D * D, _shift(l, 5), x), F, _shift(l, 6));
        unchecked {
            for (uint256 i; i < F; ++i) if (f[i] < 0) f[i] = 0;
        }
        _addTo(x, _mv(w, o + 2 * D + 4 * D * D + F * D, f, D, _shift(l, 7)));
    }

    /// Causal attention for the newest position `p`, all heads.
    function _attend(int256[] memory q, bytes memory ex, int256[] memory kc, int256[] memory vc, uint256 base, uint256 p)
        private pure returns (int256[] memory out)
    {
        out = new int256[](D);
        int256[] memory s = new int256[](p + 1);
        unchecked {
            for (uint256 hd; hd < H; ++hd) {
                uint256 c = hd * DH;
                int256 top = type(int256).min;
                for (uint256 u; u <= p; ++u) {
                    int256 dot;
                    for (uint256 i; i < DH; ++i) dot += q[c + i] * kc[base + u * D + c + i];
                    s[u] = (dot >> A) >> 2;
                    if (s[u] > top) top = s[u];
                }
                int256 den;
                for (uint256 u; u <= p; ++u) {
                    int256 e = int256(_exp(ex, uint256(top - s[u])));
                    s[u] = e;
                    den += e;
                }
                for (uint256 i; i < DH; ++i) {
                    int256 num;
                    for (uint256 u; u <= p; ++u) num += s[u] * vc[base + u * D + c + i];
                    out[c + i] = num / den;
                }
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

    /// y = (W x) >> sh, W stored as int8 rows of x.length bytes (a multiple of 32) at `off`.
    function _mv(bytes memory w, uint256 off, int256[] memory x, uint256 nout, uint256 sh)
        private pure returns (int256[] memory y)
    {
        uint256 nin = x.length;
        y = new int256[](nout);
        assembly {
            let wp := add(add(w, 32), off)
            let xp := add(x, 32)
            let yp := add(y, 32)
            for { let j := 0 } lt(j, nout) { j := add(j, 1) } {
                let acc := 0
                let rp := add(wp, mul(j, nin))
                for { let i := 0 } lt(i, nin) { i := add(i, 32) } {
                    let word := mload(add(rp, i))
                    let xq := add(xp, shl(5, i))
                    // 32 weights per word, unrolled
                    acc := add(acc, mul(mload(xq), signextend(0, byte(0, word))))
                    acc := add(acc, mul(mload(add(xq, 0x20)), signextend(0, byte(1, word))))
                    acc := add(acc, mul(mload(add(xq, 0x40)), signextend(0, byte(2, word))))
                    acc := add(acc, mul(mload(add(xq, 0x60)), signextend(0, byte(3, word))))
                    acc := add(acc, mul(mload(add(xq, 0x80)), signextend(0, byte(4, word))))
                    acc := add(acc, mul(mload(add(xq, 0xa0)), signextend(0, byte(5, word))))
                    acc := add(acc, mul(mload(add(xq, 0xc0)), signextend(0, byte(6, word))))
                    acc := add(acc, mul(mload(add(xq, 0xe0)), signextend(0, byte(7, word))))
                    acc := add(acc, mul(mload(add(xq, 0x100)), signextend(0, byte(8, word))))
                    acc := add(acc, mul(mload(add(xq, 0x120)), signextend(0, byte(9, word))))
                    acc := add(acc, mul(mload(add(xq, 0x140)), signextend(0, byte(10, word))))
                    acc := add(acc, mul(mload(add(xq, 0x160)), signextend(0, byte(11, word))))
                    acc := add(acc, mul(mload(add(xq, 0x180)), signextend(0, byte(12, word))))
                    acc := add(acc, mul(mload(add(xq, 0x1a0)), signextend(0, byte(13, word))))
                    acc := add(acc, mul(mload(add(xq, 0x1c0)), signextend(0, byte(14, word))))
                    acc := add(acc, mul(mload(add(xq, 0x1e0)), signextend(0, byte(15, word))))
                    acc := add(acc, mul(mload(add(xq, 0x200)), signextend(0, byte(16, word))))
                    acc := add(acc, mul(mload(add(xq, 0x220)), signextend(0, byte(17, word))))
                    acc := add(acc, mul(mload(add(xq, 0x240)), signextend(0, byte(18, word))))
                    acc := add(acc, mul(mload(add(xq, 0x260)), signextend(0, byte(19, word))))
                    acc := add(acc, mul(mload(add(xq, 0x280)), signextend(0, byte(20, word))))
                    acc := add(acc, mul(mload(add(xq, 0x2a0)), signextend(0, byte(21, word))))
                    acc := add(acc, mul(mload(add(xq, 0x2c0)), signextend(0, byte(22, word))))
                    acc := add(acc, mul(mload(add(xq, 0x2e0)), signextend(0, byte(23, word))))
                    acc := add(acc, mul(mload(add(xq, 0x300)), signextend(0, byte(24, word))))
                    acc := add(acc, mul(mload(add(xq, 0x320)), signextend(0, byte(25, word))))
                    acc := add(acc, mul(mload(add(xq, 0x340)), signextend(0, byte(26, word))))
                    acc := add(acc, mul(mload(add(xq, 0x360)), signextend(0, byte(27, word))))
                    acc := add(acc, mul(mload(add(xq, 0x380)), signextend(0, byte(28, word))))
                    acc := add(acc, mul(mload(add(xq, 0x3a0)), signextend(0, byte(29, word))))
                    acc := add(acc, mul(mload(add(xq, 0x3c0)), signextend(0, byte(30, word))))
                    acc := add(acc, mul(mload(add(xq, 0x3e0)), signextend(0, byte(31, word))))
                }
                mstore(add(yp, shl(5, j)), sar(sh, acc))
            }
        }
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

    function _text(uint256[] memory toks) private pure returns (bytes memory out) {
        bytes memory voc = VOCAB;
        bytes memory off = VOFF;
        for (uint256 k = 1; k < toks.length; ++k) {
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

    function _u16(bytes memory b, uint256 i) private pure returns (uint256) {
        return (uint256(uint8(b[2 * i])) << 8) | uint8(b[2 * i + 1]);
    }

    /// Per-layer shifts packed one byte each: [g1, Wq, Wk, Wv, Wo, g2, W1, W2].
    function _shift(uint256 layer, uint256 k) private pure returns (uint256) {
        return (SHIFTS >> ((layer * 8 + k) * 8)) & 0xff;
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
