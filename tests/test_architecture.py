"""Structural tests: the properties that must hold regardless of what any
individual function does.

These are what actually keep the risk gate non-bypassable over time — not the
docstrings, which erode. If someone adds a new call path to the broker's order
methods that skips `RiskEngine`, this test fails CI.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "aitrader"

#: The only file allowed to import ib_async or call its order-placement methods.
ALLOWED_IB_ASYNC_FILE = SRC / "broker" / "ib_adapter.py"

#: The only file allowed to call the order manager's mutating methods.
ALLOWED_ORDER_MANAGER_CALLERS = {
    SRC / "risk" / "engine.py",
    SRC / "engine" / "scheduler.py",  # flatten-on-kill-switch / EOD flatten paths
    SRC / "engine" / "cycle.py",  # CLOSE proposals: bypass sizing on purpose, but still
                                   # gated by RiskEngine.evaluate_close() first
    SRC / "broker" / "orders.py",  # the module itself
}

ORDER_MUTATING_METHODS = {"place_bracket", "place_single", "close_position", "cancel_all_working"}


def _iter_python_files():
    return sorted(p for p in SRC.rglob("*.py") if p.is_file())


def _imports_ib_async(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "ib_async" for alias in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "ib_async"
        ):
            return True
    return False


def _calls_place_order(tree: ast.AST) -> list[int]:
    """Line numbers of any `.placeOrder(` call — the raw ib_async primitive."""
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "placeOrder"
        ):
            lines.append(node.lineno)
    return lines


class TestIbAsyncIsolation:
    """`ib_async` must be reachable from exactly one file.

    This is what makes it possible to swap the library later, and — more
    importantly for safety — it means every place that could place a raw order
    is one file we can audit by hand.
    """

    def test_only_the_adapter_imports_ib_async(self):
        offenders = []
        for path in _iter_python_files():
            if path == ALLOWED_IB_ASYNC_FILE:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            if _imports_ib_async(tree):
                offenders.append(str(path.relative_to(SRC)))
        assert not offenders, f"ib_async imported outside the adapter: {offenders}"

    def test_only_the_adapter_calls_place_order(self):
        offenders = []
        for path in _iter_python_files():
            tree = ast.parse(path.read_text(), filename=str(path))
            lines = _calls_place_order(tree)
            if lines and path != ALLOWED_IB_ASYNC_FILE:
                offenders.append(f"{path.relative_to(SRC)}:{lines}")
        assert not offenders, f".placeOrder() called outside the adapter: {offenders}"


class TestOrderManagerBoundary:
    """The order manager's mutating methods must only be called from the small,
    explicit set of modules that are allowed to originate an order: the risk
    gate (the normal path), and the engine's flatten/close paths, which exist
    specifically to exit risk rather than take on new risk."""

    def test_order_mutating_calls_are_confined(self):
        offenders = []
        for path in _iter_python_files():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr not in ORDER_MUTATING_METHODS:
                    continue
                # A call on `self.` inside orders.py itself, or on an
                # `order_manager`/`orders` attribute elsewhere.
                if path not in ALLOWED_ORDER_MANAGER_CALLERS:
                    offenders.append(
                        f"{path.relative_to(SRC)}:{node.lineno} calls .{node.func.attr}(...)"
                    )
        assert not offenders, f"order-mutating call outside the approved boundary: {offenders}"


class TestDomainPurity:
    """`domain/` holds only data. If it starts importing infrastructure, the
    boundary between "what a trade is" and "how a trade gets placed" has
    started to blur."""

    def test_domain_imports_no_infrastructure(self):
        domain = SRC / "domain"
        forbidden_roots = {"broker", "llm", "web", "engine"}
        offenders = []
        for path in sorted(domain.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                mod = None
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module
                elif isinstance(node, ast.Import):
                    mod = node.names[0].name if node.names else None
                if not mod:
                    continue
                parts = mod.split(".")
                if parts and parts[0] == "aitrader" and len(parts) > 1 and parts[1] in forbidden_roots:
                    offenders.append(f"{path.relative_to(SRC)} imports {mod}")
        assert not offenders, f"domain/ depends on infrastructure: {offenders}"


class TestProposalCarriesNoExecutableFields:
    """Re-asserted here as a structural check, not just a unit test: the model's
    output type must never grow a field that could become a broker call
    directly (a price, a quantity, an order id)."""

    def test_trade_proposal_has_no_forbidden_fields(self):
        from aitrader.domain.proposals import TradeProposal

        forbidden = {
            "quantity", "shares", "limit_price", "stop_price", "target_price",
            "price", "order_id", "order_ref", "notional", "account_id",
        }
        present = forbidden & set(TradeProposal.model_fields)
        assert not present, f"TradeProposal carries forbidden fields: {present}"
