# tests/test_integration.py
"""Integration smoke tests — verify main.py wiring for new services."""

import importlib
import inspect
from unittest.mock import MagicMock

import pytest


def test_main_imports_strategy_engine():
    """main.py should import StrategyEngine without error."""
    import main
    # Verify the name is accessible at module level
    assert hasattr(main, 'StrategyEngine'), (
        "StrategyEngine not imported in main.py"
    )


def test_main_imports_settlement_service():
    """main.py should import SettlementService without error."""
    import main
    assert hasattr(main, 'SettlementService'), (
        "SettlementService not imported in main.py"
    )


def test_main_imports_default_db_path():
    """main.py should import DEFAULT_DB_PATH from core.db."""
    import main
    assert hasattr(main, 'DEFAULT_DB_PATH'), (
        "DEFAULT_DB_PATH not imported in main.py"
    )


def test_strategy_engine_in_main_function():
    """main() function source should reference StrategyEngine instantiation."""
    import main
    source = inspect.getsource(main.main)
    assert 'StrategyEngine(' in source, (
        "StrategyEngine not instantiated in main()"
    )
    assert 'strategy.run()' in source, (
        "strategy.run() not in task list"
    )


def test_settlement_service_in_main_function():
    """main() function source should reference SettlementService instantiation."""
    import main
    source = inspect.getsource(main.main)
    assert 'SettlementService(' in source, (
        "SettlementService not instantiated in main()"
    )
    assert 'settlement.run()' in source, (
        "settlement.run() not in task list"
    )


def test_strategy_engine_receives_paper_trader():
    """StrategyEngine should be wired with paper_trader in main()."""
    import main
    source = inspect.getsource(main.main)
    assert 'paper_trader=paper_trader' in source or \
           'StrategyEngine(db_path=DEFAULT_DB_PATH, paper_trader=paper_trader)' in source, (
        "StrategyEngine not wired with paper_trader"
    )


def test_settlement_service_receives_paper_trader():
    """SettlementService should be wired with paper_trader in main()."""
    import main
    source = inspect.getsource(main.main)
    assert 'paper_trader=paper_trader' in source or \
           'SettlementService(db_path=DEFAULT_DB_PATH, paper_trader=paper_trader)' in source, (
        "SettlementService not wired with paper_trader"
    )


def test_strategy_engine_class_has_run():
    """StrategyEngine must expose an async run() method."""
    from services.strategy_engine import StrategyEngine
    assert hasattr(StrategyEngine, 'run'), "StrategyEngine missing run() method"
    assert inspect.iscoroutinefunction(StrategyEngine.run), (
        "StrategyEngine.run() must be async"
    )


def test_settlement_service_class_has_run():
    """SettlementService must expose an async run() method."""
    from services.settlement import SettlementService
    assert hasattr(SettlementService, 'run'), "SettlementService missing run() method"
    assert inspect.iscoroutinefunction(SettlementService.run), (
        "SettlementService.run() must be async"
    )
