// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

// Outer-product packed matvec: one 32-byte word = 32 offset weights (w + 128) of one input column for 32 output rows.
// Four byte phases are masked into 8 lanes of 32 bits and multiplied by the (signed, <= 2^B) activation.
contract Kernel {
    function mv(bytes memory w, int256[] memory x, uint256 blocks) public view returns (int256[] memory y, uint256 g) {
        g = gasleft();
        uint256 nin = x.length;
        y = new int256[](blocks * 32);
        assembly {
            let M := 0x000000ff000000ff000000ff000000ff000000ff000000ff000000ff000000ff
            let wp := add(w, 32)
            let xs := add(x, 32)
            let xe := add(xs, shl(5, nin))
            let yp := add(y, 32)
            let sx := 0
            for { let p := xs } lt(p, xe) { p := add(p, 32) } { sx := add(sx, mload(p)) }
            let corr := shl(7, sx)
            for { let b := 0 } lt(b, blocks) { b := add(b, 1) } {
                let a0 := 0
                let a1 := 0
                let a2 := 0
                let a3 := 0
                let W := 0
                let v := 0
                for { let p := xs } lt(p, xe) { p := add(p, 128) } {
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
                // lanes are signed 32-bit: peel the low lane, remove it, shift down
                for { let L := 0 } lt(L, 8) { L := add(L, 1) } {
                    let v0 := signextend(3, a0)
                    mstore(add(yp, shl(5, L)), sub(v0, corr))
                    a0 := sar(32, sub(a0, v0))
                    let v1 := signextend(3, a1)
                    mstore(add(yp, shl(5, add(L, 8))), sub(v1, corr))
                    a1 := sar(32, sub(a1, v1))
                    let v2 := signextend(3, a2)
                    mstore(add(yp, shl(5, add(L, 16))), sub(v2, corr))
                    a2 := sar(32, sub(a2, v2))
                    let v3 := signextend(3, a3)
                    mstore(add(yp, shl(5, add(L, 24))), sub(v3, corr))
                    a3 := sar(32, sub(a3, v3))
                }
                yp := add(yp, 1024)
            }
        }
        g -= gasleft();
    }
}
