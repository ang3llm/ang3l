// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

// Runs against Robinhood Chain itself, deploying exactly the way launch night does:
//   HOOK_SALT=0x… forge test --match-path test/Ang3lHook.t.sol --fork-url https://4663.rpc.thirdweb.com
import {Ang3lHook, Ang3lToken, IERC20Min, IWETH} from "../src/Ang3lHook.sol";
import {Ang3lRouter} from "../src/Ang3lRouter.sol";
import {IPoolManager} from "v4-core/src/interfaces/IPoolManager.sol";
import {IUnlockCallback} from "v4-core/src/interfaces/callback/IUnlockCallback.sol";
import {TickMath} from "v4-core/src/libraries/TickMath.sol";
import {PoolKey} from "v4-core/src/types/PoolKey.sol";
import {Currency} from "v4-core/src/types/Currency.sol";
import {BalanceDelta} from "v4-core/src/types/BalanceDelta.sol";
import {ModifyLiquidityParams, SwapParams} from "v4-core/src/types/PoolOperation.sol";

interface Vm {
    function deal(address who, uint256 amount) external;
    function prank(address who) external;
    function startPrank(address who) external;
    function stopPrank() external;
    function expectRevert() external;
    function envBytes32(string calldata name) external view returns (bytes32);
}

interface IPermit2 {
    function approve(address token, address spender, uint160 amount, uint48 expiration) external;
}

interface IUniversalRouter {
    function execute(bytes calldata commands, bytes[] calldata inputs, uint256 deadline) external payable;
}

interface IV4Quoter {
    struct QuoteExactSingleParams {
        PoolKey poolKey;
        bool zeroForOne;
        uint128 exactAmount;
        bytes hookData;
    }

    function quoteExactInputSingle(QuoteExactSingleParams memory params)
        external
        returns (uint256 amountOut, uint256 gasEstimate);
}

/// A bare v4 caller: swaps, and tries the liquidity and pool moves the hook must refuse.
contract Caller is IUnlockCallback {
    IPoolManager immutable pm;

    constructor(IPoolManager pm_) {
        pm = pm_;
    }

    function swap(PoolKey memory key, bool zeroForOne, int256 amount) external returns (BalanceDelta) {
        return abi.decode(pm.unlock(abi.encode(uint8(0), abi.encode(key, zeroForOne, amount))), (BalanceDelta));
    }

    function modify(PoolKey memory key, int256 liquidity) external {
        pm.unlock(abi.encode(uint8(1), abi.encode(key, liquidity)));
    }

    function unlockCallback(bytes calldata data) external returns (bytes memory) {
        (uint8 op, bytes memory args) = abi.decode(data, (uint8, bytes));
        if (op == 0) {
            (PoolKey memory key, bool zeroForOne, int256 amount) = abi.decode(args, (PoolKey, bool, int256));
            uint160 limit = zeroForOne ? TickMath.MIN_SQRT_PRICE + 1 : TickMath.MAX_SQRT_PRICE - 1;
            BalanceDelta d = pm.swap(key, SwapParams(zeroForOne, amount, limit), "");
            _close(key.currency0, d.amount0());
            _close(key.currency1, d.amount1());
            return abi.encode(d);
        }
        (PoolKey memory k, int256 liq) = abi.decode(args, (PoolKey, int256));
        pm.modifyLiquidity(k, ModifyLiquidityParams(-887_200, 201_600, liq, 0), "");
        return "";
    }

    function _close(Currency c, int128 amount) private {
        if (amount < 0) {
            pm.sync(c);
            IERC20Min(Currency.unwrap(c)).transfer(address(pm), uint128(-amount));
            pm.settle();
        } else if (amount > 0) {
            pm.take(c, address(this), uint128(amount));
        }
    }

    receive() external payable {}
}

contract Ang3lHookTest {
    Vm constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));
    IPoolManager constant PM = IPoolManager(0x8366a39CC670B4001A1121B8F6A443A643e40951);
    address constant WETH = 0x0Bd7D308f8E1639FAb988df18A8011f41EAcAD73;
    address constant DEPLOYER = 0x4e59b44847b379578588920cA78FbF26c0B4956C; // the CREATE2 proxy launch night uses
    address constant ADMIN = 0x7Cfd8b2f3EF62656285E508F3A11745DeC8A92Fd;
    IPermit2 constant PERMIT2 = IPermit2(0x000000000022D473030F116dDEE9F6B43aC78BA3);
    IUniversalRouter constant UR = IUniversalRouter(0x204FAca1764B154221e35c0d20aBb3c525710498);
    IV4Quoter constant QUOTER = IV4Quoter(0x8Dc178eFB8111BB0973Dd9d722ebeFF267c98F94);

    Ang3lHook hook;
    Ang3lToken token;
    Caller caller;
    PoolKey key;

    function setUp() public {
        bytes memory init = abi.encodePacked(type(Ang3lHook).creationCode, abi.encode(PM, WETH, ADMIN, ADMIN));
        (bool ok, bytes memory ret) = DEPLOYER.call(abi.encodePacked(vm.envBytes32("HOOK_SALT"), init));
        require(ok && ret.length == 20, "deploy failed");
        hook = Ang3lHook(payable(address(bytes20(ret))));
        token = hook.token();
        key = hook.poolKey();
        caller = new Caller(PM);
        vm.deal(ADMIN, 1 ether);
        vm.prank(ADMIN);
        hook.launch{value: 0.03 ether}();
    }

    // ---- launch ----

    function test_launch_gives_admin_a_fee_free_first_buy() public view {
        uint256 got = token.balanceOf(ADMIN);
        // ~1.76 ETH fully diluted: 0.03 ETH buys about 1.67% of the supply
        require(got > 16_000_000e18 && got < 17_000_000e18, "dev buy size");
        require(hook.pending() == 0, "dev buy paid a fee");
        require(token.balanceOf(address(hook)) < 1e6, "supply not pooled");
        require(hook.launched(), "flag");
    }

    function test_launch_only_once_and_only_admin() public {
        vm.deal(address(this), 1 ether);
        vm.expectRevert();
        hook.launch{value: 0.01 ether}();
        vm.prank(ADMIN);
        vm.expectRevert();
        hook.launch();
    }

    // ---- the 2% fee, all four swap shapes ----

    function test_buy_exact_in_pays_2pct_of_weth_in() public {
        _weth(address(caller), 1 ether);
        BalanceDelta d = caller.swap(key, true, -1 ether);
        require(d.amount0() == -1 ether, "buyer pays exactly the input");
        require(hook.pending() == 0.02 ether, "fee is 2% of the input");
        require(d.amount1() > 0, "tokens out");
    }

    function test_sell_exact_in_pays_2pct_of_weth_out() public {
        _weth(address(caller), 1 ether);
        caller.swap(key, true, -1 ether);
        uint256 before = hook.pending();
        BalanceDelta d = caller.swap(key, false, -int256(token.balanceOf(address(caller)) / 2));
        uint256 fee = hook.pending() - before;
        uint256 net = uint128(d.amount0());
        // net = 98% of the gross, fee = 2% of it: net is 49x the fee, within rounding
        require(fee > 0 && _near(net, fee * 49, 49), "fee is 2% of the WETH out");
    }

    function test_buy_exact_out_pays_2pct_on_top() public {
        _weth(address(caller), 1 ether);
        BalanceDelta d = caller.swap(key, true, 1_000_000e18);
        require(d.amount1() == 1_000_000e18, "exact tokens out");
        uint256 paid = uint128(-d.amount0());
        // paid = 102% of what the pool took, fee = 2% of it: paid is 51x the fee, within rounding
        require(_near(paid, hook.pending() * 51, 51), "fee is 2% of the WETH in");
    }

    function test_sell_exact_out_pays_2pct_of_weth_out() public {
        _weth(address(caller), 1 ether);
        caller.swap(key, true, -1 ether);
        uint256 before = hook.pending();
        BalanceDelta d = caller.swap(key, false, 0.1 ether);
        require(d.amount0() == 0.1 ether, "exact WETH out");
        require(hook.pending() - before == 0.002 ether, "fee is 2% of the WETH out");
    }

    // ---- liquidity is locked, the hook is closed to other pools ----

    function test_nobody_can_remove_or_add_liquidity() public {
        vm.expectRevert();
        caller.modify(key, -1);
        vm.expectRevert();
        caller.modify(key, 1e18);
    }

    function test_no_other_pool_on_this_hook() public {
        PoolKey memory other = key;
        other.tickSpacing = 60;
        vm.expectRevert();
        PM.initialize(other, TickMath.getSqrtPriceAtTick(0));
    }

    // ---- fees out ----

    function test_collect_sends_eth_to_fee_wallet() public {
        _weth(address(caller), 1 ether);
        caller.swap(key, true, -1 ether);
        caller.swap(key, false, -int256(token.balanceOf(address(caller))));
        uint256 fees = hook.pending();
        uint256 before = ADMIN.balance;
        hook.collect(); // anyone may push them
        require(ADMIN.balance - before == fees && hook.pending() == 0, "collected");
    }

    function test_only_fee_wallet_moves_fee_wallet() public {
        vm.expectRevert();
        hook.setFeeTo(address(this));
        vm.prank(ADMIN);
        hook.setFeeTo(address(this));
        require(hook.feeTo() == address(this), "moved");
    }

    // ---- the real Universal Router and quoter, as wallets and terminals use them ----

    function test_universal_router_buy_and_sell() public {
        address user = address(0xB0B);
        _weth(user, 1 ether);
        vm.startPrank(user);
        IERC20Approve(WETH).approve(address(PERMIT2), type(uint256).max);
        PERMIT2.approve(WETH, address(UR), type(uint160).max, type(uint48).max);
        (uint256 quoted,) = QUOTER.quoteExactInputSingle(IV4Quoter.QuoteExactSingleParams(key, true, 0.5 ether, ""));
        _ur(true, 0.5 ether, WETH, address(token));
        uint256 got = token.balanceOf(user);
        require(got == quoted && got > 0, "router buy matches the quote");
        require(hook.pending() == 0.01 ether, "router buy paid 2%");

        IERC20Approve(address(token)).approve(address(PERMIT2), type(uint256).max);
        PERMIT2.approve(address(token), address(UR), type(uint160).max, type(uint48).max);
        uint256 wethBefore = IERC20Min(WETH).balanceOf(user);
        _ur(false, uint128(got), address(token), WETH);
        uint256 net = IERC20Min(WETH).balanceOf(user) - wethBefore;
        require(net > 0.48 ether && net < 0.5 ether, "router sell returns ~2% less twice");
        require(_near(net, (hook.pending() - 0.01 ether) * 49, 49), "router sell paid 2%");
        vm.stopPrank();
    }

    // ---- the site's buy box: plain ETH in and out, one call each ----

    function test_ang3l_router_buy_and_sell_with_eth() public {
        Ang3lRouter router = new Ang3lRouter(hook);
        address user = address(0xCAFE);
        vm.deal(user, 1 ether);
        vm.startPrank(user);
        (uint256 quoted,) = QUOTER.quoteExactInputSingle(IV4Quoter.QuoteExactSingleParams(key, true, 0.5 ether, ""));
        vm.expectRevert();
        router.buy{value: 0.5 ether}(quoted + 1); // slippage guard
        uint256 got = router.buy{value: 0.5 ether}(quoted);
        require(got == quoted && token.balanceOf(user) == got, "buy matches the quote");
        require(hook.pending() == 0.01 ether && user.balance == 0.5 ether, "buy paid 2%");

        token.approve(address(router), got);
        uint256 out = router.sell(got, 0.48 ether);
        require(user.balance == 0.5 ether + out && token.balanceOf(user) == 0, "sold for ETH");
        require(_near(out, (hook.pending() - 0.01 ether) * 49, 49), "sell paid 2%");
        require(address(router).balance == 0 && token.balanceOf(address(router)) == 0, "router keeps nothing");
        vm.stopPrank();
    }

    // ---- helpers ----

    struct ExactInputSingle {
        PoolKey poolKey;
        bool zeroForOne;
        uint128 amountIn;
        uint128 amountOutMinimum;
        uint256 minHopPriceX36;
        bytes hookData;
    }

    function _ur(bool zeroForOne, uint128 amountIn, address tokenIn, address tokenOut) private {
        bytes[] memory params = new bytes[](3);
        params[0] = abi.encode(ExactInputSingle(key, zeroForOne, amountIn, 1, 0, ""));
        params[1] = abi.encode(tokenIn, uint256(amountIn)); // SETTLE_ALL
        params[2] = abi.encode(tokenOut, uint256(1)); // TAKE_ALL
        bytes[] memory inputs = new bytes[](1);
        inputs[0] = abi.encode(abi.encodePacked(uint8(0x06), uint8(0x0c), uint8(0x0f)), params);
        UR.execute(abi.encodePacked(uint8(0x10)), inputs, block.timestamp + 60);
    }

    function _weth(address to, uint256 amount) private {
        vm.deal(address(this), amount);
        IWETH(WETH).deposit{value: amount}();
        IERC20Min(WETH).transfer(to, amount);
    }

    function _near(uint256 a, uint256 b, uint256 tol) private pure returns (bool) {
        return a > b ? a - b <= tol : b - a <= tol;
    }

    receive() external payable {}
}

interface IERC20Approve {
    function approve(address, uint256) external returns (bool);
}
