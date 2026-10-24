// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;
import {Kernel} from "./Kernel.sol";

contract KernelTest {
    Kernel k;
    event log_named_uint(string key, uint256 val);
    function setUp() public { k = new Kernel(); }

    // y = W x for a 256 x 256 int8 matrix, checked against the plain loop; reports gas per multiply-accumulate.
    function test_kernel() public {
        uint256 nin = 256; uint256 nout = 256;
        bytes memory w = new bytes(nin * nout);
        int8[] memory ws = new int8[](nin * nout);
        int256[] memory x = new int256[](nin);
        for (uint256 i; i < nin; ++i) x[i] = int256(uint256(keccak256(abi.encode("x", i))) % 32768) - 16384;
        for (uint256 j; j < nout; ++j) for (uint256 i; i < nin; ++i) {
            int8 v = int8(int256(uint256(keccak256(abi.encode(j, i))) % 255) - 127);
            ws[j * nin + i] = v;
            // block j/32, column i, row r = 8p + L sits at byte 31 - 4L - p
            uint256 r = j % 32;
            w[((j / 32) * nin + i) * 32 + 31 - 4 * (r % 8) - r / 8] = bytes1(uint8(int8(v) + 0) ^ 0x80);
        }
        (int256[] memory y, uint256 g) = k.mv(w, x, nout / 32);
        for (uint256 j; j < nout; ++j) {
            int256 s;
            for (uint256 i; i < nin; ++i) s += int256(ws[j * nin + i]) * x[i];
            require(y[j] == s, "mismatch");
        }
        emit log_named_uint("gas for the 256 x 256 product", g);
        emit log_named_uint("gas per multiply-accumulate x100", g * 100 / (nin * nout));
    }

}
