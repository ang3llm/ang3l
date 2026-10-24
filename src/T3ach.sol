// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

/// @title T3ach: lessons for ANG3L's next model
/// @notice Anyone may write one line a UTC day for the next AM3N to learn from. The lines live in this contract's
/// event log, signed by the wallet that wrote them, for everyone to read. Nothing else is kept and nothing is paid
/// but gas: no owner, no admin, no funds.
contract T3ach {
    /// @notice A lesson: who wrote it, its number (from 0) and the line itself.
    event Lesson(address indexed teacher, uint256 indexed n, string line);

    /// @notice How many lessons have been written.
    uint256 public lessons;
    /// @notice The UTC day (days since 1970) of a wallet's last lesson, plus one; 0 if it never taught.
    mapping(address => uint256) public taught;

    error Length();     // 8 to 96 bytes
    error Character();  // printable ASCII only, 0x20 to 0x7e: the model reads nothing else
    error Today();      // one lesson per wallet per UTC day

    function teach(string calldata line) external {
        bytes calldata b = bytes(line);
        if (b.length < 8 || b.length > 96) revert Length();
        for (uint256 i; i < b.length; ++i) if (b[i] < 0x20 || b[i] > 0x7e) revert Character();
        uint256 day = block.timestamp / 1 days + 1;
        if (taught[msg.sender] == day) revert Today();
        taught[msg.sender] = day;
        emit Lesson(msg.sender, lessons++, line);
    }
}
