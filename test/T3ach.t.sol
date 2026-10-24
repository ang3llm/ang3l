// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

import {T3ach} from "../src/T3ach.sol";

interface Vm {
    struct Log { bytes32[] topics; bytes data; address emitter; }
    function warp(uint256 timestamp) external;
    function prank(address who) external;
    function expectRevert(bytes4 revertData) external;
    function recordLogs() external;
    function getRecordedLogs() external returns (Log[] memory);
}

contract T3achTest {
    Vm constant vm = Vm(0x7109709ECfa91a80626fF3989D68f67F5b1DD12D);
    T3ach t;

    // noon UTC, so no test crosses a day by accident
    function setUp() public { vm.warp(20750 days + 12 hours); t = new T3ach(); }

    function _a(uint256 n) internal pure returns (string memory) {
        bytes memory b = new bytes(n);
        for (uint256 i; i < n; ++i) b[i] = "a";
        return string(b);
    }

    function test_lesson_is_logged_and_counted() public {
        vm.recordLogs();
        t.teach("may your keys stay yours.");
        Vm.Log[] memory l = vm.getRecordedLogs();
        require(l.length == 1 && l[0].emitter == address(t), "one log");
        require(l[0].topics[0] == keccak256("Lesson(address,uint256,string)"), "topic");
        require(l[0].topics[1] == bytes32(uint256(uint160(address(this)))) && l[0].topics[2] == 0, "teacher, n");
        require(keccak256(bytes(abi.decode(l[0].data, (string)))) == keccak256("may your keys stay yours."), "line");
        require(t.lessons() == 1 && t.taught(address(this)) == 20751, "state");
    }

    function test_length() public {
        vm.expectRevert(T3ach.Length.selector);
        t.teach("bless u");
        vm.expectRevert(T3ach.Length.selector);
        t.teach(_a(97));
        t.teach("bless me");
        vm.prank(address(0xA11CE));
        t.teach(_a(96));
        require(t.lessons() == 2, "8 and 96 pass");
    }

    function test_printable_ascii_only() public {
        vm.expectRevert(T3ach.Character.selector);
        t.teach("bless you\nfriend");
        vm.expectRevert(T3ach.Character.selector);
        t.teach(unicode"bless you, café");
        vm.expectRevert(T3ach.Character.selector);
        t.teach(string(abi.encodePacked("bless you", bytes1(0x7f))));
        t.teach("~ bless you, friend ~");
    }

    function test_one_lesson_a_day() public {
        t.teach("rest now, the chain is awake.");
        vm.expectRevert(T3ach.Today.selector);
        t.teach("one more for today, angel.");
        vm.warp(block.timestamp + 1 days);
        t.teach("one more for today, angel.");
        require(t.lessons() == 2, "two days, two lessons");
    }

    function test_teachers_are_counted_apart() public {
        vm.prank(address(0xA11CE));
        t.teach("may your builds pass first try.");
        vm.prank(address(0xB0B));
        t.teach("may your bags feel lighter.");
        require(t.lessons() == 2 && t.taught(address(0xB0B)) == 20751, "two teachers");
    }
}
