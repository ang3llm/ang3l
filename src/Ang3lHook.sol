// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

import {IHooks} from "v4-core/src/interfaces/IHooks.sol";
import {IPoolManager} from "v4-core/src/interfaces/IPoolManager.sol";
import {IUnlockCallback} from "v4-core/src/interfaces/callback/IUnlockCallback.sol";
import {Hooks} from "v4-core/src/libraries/Hooks.sol";
import {TickMath} from "v4-core/src/libraries/TickMath.sol";
import {FullMath} from "v4-core/src/libraries/FullMath.sol";
import {FixedPoint96} from "v4-core/src/libraries/FixedPoint96.sol";
import {PoolKey} from "v4-core/src/types/PoolKey.sol";
import {Currency} from "v4-core/src/types/Currency.sol";
import {BalanceDelta} from "v4-core/src/types/BalanceDelta.sol";
import {BeforeSwapDelta, toBeforeSwapDelta} from "v4-core/src/types/BeforeSwapDelta.sol";
import {ModifyLiquidityParams, SwapParams} from "v4-core/src/types/PoolOperation.sol";

interface IERC20Min {
    function balanceOf(address) external view returns (uint256);
    function transfer(address, uint256) external returns (bool);
}

interface IWETH is IERC20Min {
    function deposit() external payable;
    function withdraw(uint256) external;
}

/// @title $ANG3L
/// @notice A plain ERC-20: the whole supply is minted once, to the hook that pools it. No owner, no tax, no mint.
contract Ang3lToken {
    string public constant name = "ANG3L";
    string public constant symbol = "ANG3L";
    uint8 public constant decimals = 18;
    uint256 public constant totalSupply = 1_000_000_000e18;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor() {
        balanceOf[msg.sender] = totalSupply;
        emit Transfer(address(0), msg.sender, totalSupply);
    }

    function approve(address spender, uint256 value) external returns (bool) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transfer(address to, uint256 value) external returns (bool) {
        _move(msg.sender, to, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) external returns (bool) {
        uint256 allowed = allowance[from][msg.sender];
        if (allowed != type(uint256).max) allowance[from][msg.sender] = allowed - value; // reverts when short
        _move(from, to, value);
        return true;
    }

    function _move(address from, address to, uint256 value) private {
        balanceOf[from] -= value; // reverts when short
        unchecked {
            balanceOf[to] += value; // balances sum to the fixed supply, so this never overflows
        }
        emit Transfer(from, to, value);
    }
}

/// @title The one pool $ANG3L trades in
/// @notice Uniswap v4, WETH / ANG3L, 0% LP fee. This hook:
///  - takes 2% of the WETH side of every swap for the creator, kept as PoolManager claims until collect();
///  - holds the pool's only liquidity, the whole supply, and nobody can ever remove it;
///  - lets nobody else add liquidity or open another pool on this hook.
/// Its address must carry exactly the permission bits in FLAGS (mined with CREATE2).
contract Ang3lHook is IUnlockCallback {
    uint256 public constant FEE_BPS = 200; // 2% of the WETH side
    int24 public constant TICK_SPACING = 200;
    int24 public constant START_TICK = 201_600; // ~5.7e8 ANG3L per WETH: about 1.76 ETH fully diluted at the first trade
    int24 public constant MIN_TICK = -887_200; // lowest usable tick at this spacing
    uint160 public constant FLAGS = Hooks.BEFORE_INITIALIZE_FLAG | Hooks.BEFORE_ADD_LIQUIDITY_FLAG
        | Hooks.BEFORE_REMOVE_LIQUIDITY_FLAG | Hooks.BEFORE_SWAP_FLAG | Hooks.AFTER_SWAP_FLAG
        | Hooks.BEFORE_SWAP_RETURNS_DELTA_FLAG | Hooks.AFTER_SWAP_RETURNS_DELTA_FLAG;

    IPoolManager public immutable poolManager;
    address public immutable weth;
    address public immutable admin; // may launch, once
    Ang3lToken public immutable token;
    address public feeTo; // receives the fees; only feeTo can hand that to another address
    bool public launched;

    error NotPoolManager();
    error NotAllowed();
    error LiquidityLocked();

    event Collected(address indexed to, uint256 amount);
    event FeeToChanged(address indexed feeTo);

    constructor(IPoolManager poolManager_, address weth_, address admin_, address feeTo_) {
        if (uint160(address(this)) & Hooks.ALL_HOOK_MASK != FLAGS) revert NotAllowed();
        poolManager = poolManager_;
        weth = weth_;
        admin = admin_;
        feeTo = feeTo_;
        token = new Ang3lToken();
        // WETH sorts first, so the fee side is always currency0
        if (uint160(weth_) >= uint160(address(token))) revert NotAllowed();
    }

    function poolKey() public view returns (PoolKey memory) {
        return PoolKey(Currency.wrap(weth), Currency.wrap(address(token)), 0, TICK_SPACING, IHooks(address(this)));
    }

    /// @notice Creator fees collected so far and not yet sent, in WETH.
    function pending() public view returns (uint256) {
        return poolManager.balanceOf(address(this), uint256(uint160(weth)));
    }

    /// @notice Opens the pool with the whole supply as liquidity, then buys for the admin with msg.value (no fee).
    function launch() external payable {
        if (msg.sender != admin || launched) revert NotAllowed();
        launched = true;
        poolManager.initialize(poolKey(), TickMath.getSqrtPriceAtTick(START_TICK));
        poolManager.unlock(abi.encode(msg.value));
    }

    /// @notice Sends every fee collected so far to feeTo, as ETH. Anyone may call it.
    function collect() external {
        uint256 amount = pending();
        if (amount == 0) return;
        poolManager.unlock("");
        IWETH(weth).withdraw(amount);
        (bool ok,) = feeTo.call{value: amount}("");
        if (!ok) revert NotAllowed();
        emit Collected(feeTo, amount);
    }

    function setFeeTo(address next) external {
        if (msg.sender != feeTo || next == address(0)) revert NotAllowed();
        feeTo = next;
        emit FeeToChanged(next);
    }

    function unlockCallback(bytes calldata data) external returns (bytes memory) {
        if (msg.sender != address(poolManager)) revert NotPoolManager();
        if (data.length == 0) {
            // collect: turn the claims back into WETH held here
            uint256 amount = pending();
            poolManager.burn(address(this), uint256(uint160(weth)), amount);
            poolManager.take(Currency.wrap(weth), address(this), amount);
            return "";
        }
        // launch: one position from START_TICK down to MIN_TICK, so at the start price it is all ANG3L
        PoolKey memory key = poolKey();
        uint160 sqrtStart = TickMath.getSqrtPriceAtTick(START_TICK);
        uint256 liquidity =
            FullMath.mulDiv(token.balanceOf(address(this)), FixedPoint96.Q96, sqrtStart - TickMath.getSqrtPriceAtTick(MIN_TICK));
        (BalanceDelta added,) =
            poolManager.modifyLiquidity(key, ModifyLiquidityParams(MIN_TICK, START_TICK, int256(liquidity), 0), "");
        _pay(address(token), uint128(-added.amount1()));

        uint256 devBuy = abi.decode(data, (uint256));
        if (devBuy != 0) {
            // the hook's own swap skips its hooks, so this first buy pays no fee
            BalanceDelta bought =
                poolManager.swap(key, SwapParams(true, -int256(devBuy), TickMath.MIN_SQRT_PRICE + 1), "");
            IWETH(weth).deposit{value: devBuy}();
            _pay(weth, uint128(-bought.amount0()));
            poolManager.take(key.currency1, admin, uint128(bought.amount1()));
        }
        return "";
    }

    // ---- hook callbacks (only the ones in FLAGS are ever called) ----

    function beforeInitialize(address, PoolKey calldata, uint160) external pure returns (bytes4) {
        revert NotAllowed(); // the hook opens its own pool itself, which skips this
    }

    function beforeAddLiquidity(address, PoolKey calldata, ModifyLiquidityParams calldata, bytes calldata)
        external
        pure
        returns (bytes4)
    {
        revert NotAllowed();
    }

    function beforeRemoveLiquidity(address, PoolKey calldata, ModifyLiquidityParams calldata, bytes calldata)
        external
        pure
        returns (bytes4)
    {
        revert LiquidityLocked();
    }

    function beforeSwap(address, PoolKey calldata, SwapParams calldata params, bytes calldata)
        external
        returns (bytes4, BeforeSwapDelta, uint24)
    {
        if (msg.sender != address(poolManager)) revert NotPoolManager();
        bool exactIn = params.amountSpecified < 0;
        int128 fee;
        // WETH is the amount the swapper named when buying exact-in or selling exact-out
        if (params.zeroForOne == exactIn) {
            fee = _charge(exactIn ? uint256(-params.amountSpecified) : uint256(params.amountSpecified));
        }
        return (IHooks.beforeSwap.selector, toBeforeSwapDelta(fee, 0), 0);
    }

    function afterSwap(address, PoolKey calldata, SwapParams calldata params, BalanceDelta delta, bytes calldata)
        external
        returns (bytes4, int128)
    {
        if (msg.sender != address(poolManager)) revert NotPoolManager();
        int128 fee;
        // otherwise WETH is the side the pool worked out: what a buyer pays exact-out, or what a seller gets exact-in
        if (params.zeroForOne != (params.amountSpecified < 0)) {
            int128 side = delta.amount0();
            fee = _charge(uint128(side < 0 ? -side : side));
        }
        return (IHooks.afterSwap.selector, fee);
    }

    /// @dev Mints the fee as claims to this hook; the hook delta the PoolManager books after the callback settles it.
    function _charge(uint256 amount) private returns (int128) {
        uint256 fee = amount * FEE_BPS / 10_000;
        if (fee != 0) poolManager.mint(address(this), uint256(uint160(weth)), fee);
        return int128(int256(fee));
    }

    function _pay(address currency, uint256 amount) private {
        poolManager.sync(Currency.wrap(currency));
        if (!IERC20Min(currency).transfer(address(poolManager), amount)) revert NotAllowed();
        poolManager.settle();
    }

    receive() external payable {
        if (msg.sender != weth) revert NotAllowed(); // only WETH unwrapping in collect()
    }
}
