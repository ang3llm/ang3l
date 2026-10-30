// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

import {Ang3lHook, IERC20Min, IWETH} from "./Ang3lHook.sol";
import {IPoolManager} from "v4-core/src/interfaces/IPoolManager.sol";
import {IUnlockCallback} from "v4-core/src/interfaces/callback/IUnlockCallback.sol";
import {TickMath} from "v4-core/src/libraries/TickMath.sol";
import {Currency} from "v4-core/src/types/Currency.sol";
import {BalanceDelta} from "v4-core/src/types/BalanceDelta.sol";
import {SwapParams} from "v4-core/src/types/PoolOperation.sol";

interface IERC20From {
    function transferFrom(address, address, uint256) external returns (bool);
}

/// @title Buy and sell $ANG3L in one transaction, with plain ETH
/// @notice A thin way into the one ANG3L pool for ang3l.ink's buy box (and anyone's explorer). It holds nothing
/// between calls; the hook's 2% applies exactly as on any other route.
contract Ang3lRouter is IUnlockCallback {
    Ang3lHook public immutable hook;
    IPoolManager public immutable poolManager;
    address public immutable weth;
    address public immutable token;

    error NotAllowed();
    error TooLittle();

    constructor(Ang3lHook hook_) {
        hook = hook_;
        poolManager = hook_.poolManager();
        weth = hook_.weth();
        token = address(hook_.token());
    }

    /// @notice Buys ANG3L with all the ETH sent, for the caller. Reverts if that is fewer than minOut.
    function buy(uint256 minOut) external payable returns (uint256 out) {
        out = abi.decode(poolManager.unlock(abi.encode(msg.sender, true, msg.value)), (uint256));
        if (out < minOut) revert TooLittle();
    }

    /// @notice Sells `amount` of the caller's ANG3L (approve this router first) for ETH. Reverts below minOut.
    function sell(uint256 amount, uint256 minOut) external returns (uint256 out) {
        out = abi.decode(poolManager.unlock(abi.encode(msg.sender, false, amount)), (uint256));
        if (out < minOut) revert TooLittle();
    }

    function unlockCallback(bytes calldata data) external returns (bytes memory) {
        if (msg.sender != address(poolManager)) revert NotAllowed();
        (address user, bool isBuy, uint256 amount) = abi.decode(data, (address, bool, uint256));
        uint160 limit = isBuy ? TickMath.MIN_SQRT_PRICE + 1 : TickMath.MAX_SQRT_PRICE - 1;
        BalanceDelta d = poolManager.swap(hook.poolKey(), SwapParams(isBuy, -int256(amount), limit), "");
        uint256 out;
        if (isBuy) {
            IWETH(weth).deposit{value: amount}();
            poolManager.sync(Currency.wrap(weth));
            if (!IERC20Min(weth).transfer(address(poolManager), uint128(-d.amount0()))) revert NotAllowed();
            poolManager.settle();
            out = uint128(d.amount1());
            poolManager.take(Currency.wrap(token), user, out);
        } else {
            poolManager.sync(Currency.wrap(token));
            if (!IERC20From(token).transferFrom(user, address(poolManager), uint128(-d.amount1()))) revert NotAllowed();
            poolManager.settle();
            out = uint128(d.amount0());
            poolManager.take(Currency.wrap(weth), address(this), out);
            IWETH(weth).withdraw(out);
            (bool ok,) = user.call{value: out}("");
            if (!ok) revert NotAllowed();
        }
        return abi.encode(out);
    }

    receive() external payable {
        if (msg.sender != weth) revert NotAllowed(); // only WETH unwrapping in sell()
    }
}
