"""Configuration: YAML for strategy, environment for secrets.

The paper/live interlock lives here. Going live requires three independent
things to agree, and any disagreement is a startup abort rather than a runtime
warning. Defaults everywhere are paper.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import hard_limits as HL  # noqa: N812 - conventional shorthand, used throughout this file
from .domain.enums import KillSwitchAction, TradingMode

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


class LiveTradingNotPermitted(RuntimeError):
    """Raised at startup when the live-trading interlock is not fully satisfied."""


# --------------------------------------------------------------------------- #
# YAML-backed strategy configuration
# --------------------------------------------------------------------------- #


class RiskConfig(BaseModel):
    """Every number the risk gate enforces. Deliberately conservative by default."""

    #: Fraction of equity risked on a single trade at full conviction.
    max_risk_per_trade_pct: float = Field(default=0.005, gt=0, le=0.05)
    #: Floor on risk fraction, used at minimum actionable conviction.
    min_risk_per_trade_pct: float = Field(default=0.0015, gt=0, le=0.05)
    #: Cap on a single position's notional as a fraction of equity.
    max_position_notional_pct: float = Field(default=0.10, gt=0, le=1.0)
    max_concurrent_positions: int = Field(default=5, ge=1, le=50)
    max_new_entries_per_cycle: int = Field(default=2, ge=1, le=20)
    #: Halt for the day when total P&L drops below this fraction of starting equity.
    daily_loss_limit_pct: float = Field(default=0.02, gt=0, le=0.5)
    max_trades_per_day: int = Field(default=40, ge=1)
    #: Reject a limit price further than this fraction from last trade.
    price_collar_pct: float = Field(default=0.02, gt=0, le=0.5)
    max_spread_pct: float = Field(default=0.005, gt=0, le=0.1)
    min_avg_volume: float = Field(default=500_000, ge=0)
    min_price: float = Field(default=3.0, ge=0)
    max_price: float = Field(default=2000.0, gt=0)
    #: Reject when the latest quote is older than this.
    max_quote_age_seconds: float = Field(default=15.0, gt=0)
    #: Reject when the account snapshot is older than this.
    max_account_age_seconds: float = Field(default=120.0, gt=0)
    #: Fraction of buying power held back as a buffer.
    buying_power_reserve_pct: float = Field(default=0.20, ge=0, lt=1.0)
    #: No new entry in the same symbol within this window.
    symbol_cooldown_seconds: float = Field(default=300.0, ge=0)
    #: No reversal of a position closed within this window.
    flip_flop_guard_seconds: float = Field(default=600.0, ge=0)
    #: No new entries in the last N minutes before the close.
    no_entry_minutes_before_close: int = Field(default=20, ge=0)
    #: Flatten everything this many minutes before the close.
    flatten_minutes_before_close: int = Field(default=10, ge=0)
    max_orders_per_minute: int = Field(default=6, ge=1)
    #: Reject an LLM decision older than this by the time it reaches execution.
    max_decision_age_seconds: float = Field(default=120.0, gt=0)
    allow_shorts: bool = False
    require_risk_officer: bool = True
    kill_switch_action: KillSwitchAction = KillSwitchAction.HALT_NEW_ENTRIES

    #: Populated by the clamping validator; logged loudly at startup.
    clamp_warnings: list[str] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def _check_ordering(self) -> RiskConfig:
        if self.min_risk_per_trade_pct > self.max_risk_per_trade_pct:
            raise ValueError("min_risk_per_trade_pct must be <= max_risk_per_trade_pct")
        if self.flatten_minutes_before_close > self.no_entry_minutes_before_close:
            raise ValueError("flatten_minutes_before_close must be <= no_entry_minutes_before_close")
        if self.min_price >= self.max_price:
            raise ValueError("min_price must be < max_price")
        return self

    @model_validator(mode="after")
    def _clamp_to_hard_limits(self) -> RiskConfig:
        """Cap every field at its absolute ceiling.

        Config may tighten a limit but never loosen it. Anything clamped is
        recorded so `main` can log it at startup — a silent clamp would be worse
        than the mistake it corrects.
        """
        w: list[str] = []
        self.max_risk_per_trade_pct = HL.clamp(
            self.max_risk_per_trade_pct, HL.ABS_MAX_RISK_PER_TRADE_PCT, "max_risk_per_trade_pct", w
        )
        self.min_risk_per_trade_pct = min(self.min_risk_per_trade_pct, self.max_risk_per_trade_pct)
        self.max_position_notional_pct = HL.clamp(
            self.max_position_notional_pct,
            HL.ABS_MAX_POSITION_NOTIONAL_PCT,
            "max_position_notional_pct",
            w,
        )
        self.max_concurrent_positions = HL.clamp_int(
            self.max_concurrent_positions,
            HL.ABS_MAX_CONCURRENT_POSITIONS,
            "max_concurrent_positions",
            w,
        )
        self.max_new_entries_per_cycle = HL.clamp_int(
            self.max_new_entries_per_cycle,
            HL.ABS_MAX_NEW_ENTRIES_PER_CYCLE,
            "max_new_entries_per_cycle",
            w,
        )
        self.daily_loss_limit_pct = HL.clamp(
            self.daily_loss_limit_pct, HL.ABS_MAX_DAILY_LOSS_PCT, "daily_loss_limit_pct", w
        )
        self.max_trades_per_day = HL.clamp_int(
            self.max_trades_per_day, HL.ABS_MAX_TRADES_PER_DAY, "max_trades_per_day", w
        )
        self.max_orders_per_minute = HL.clamp_int(
            self.max_orders_per_minute, HL.ABS_MAX_ORDERS_PER_MINUTE, "max_orders_per_minute", w
        )
        self.price_collar_pct = HL.clamp(
            self.price_collar_pct, HL.ABS_MAX_PRICE_COLLAR_PCT, "price_collar_pct", w
        )
        self.max_spread_pct = HL.clamp(
            self.max_spread_pct, HL.ABS_MAX_SPREAD_PCT, "max_spread_pct", w
        )
        self.max_quote_age_seconds = HL.clamp(
            self.max_quote_age_seconds, HL.ABS_MAX_QUOTE_AGE_SECONDS, "max_quote_age_seconds", w
        )
        self.max_account_age_seconds = HL.clamp(
            self.max_account_age_seconds,
            HL.ABS_MAX_ACCOUNT_AGE_SECONDS,
            "max_account_age_seconds",
            w,
        )
        self.max_decision_age_seconds = HL.clamp(
            self.max_decision_age_seconds,
            HL.ABS_MAX_DECISION_AGE_SECONDS,
            "max_decision_age_seconds",
            w,
        )
        self.clamp_warnings = w
        return self


class UniverseConfig(BaseModel):
    symbols_file: str = "config/universe.csv"
    #: How many names get live streaming subscriptions.
    focus_list_size: int = Field(default=20, ge=1, le=80)
    #: A challenger must beat the incumbent's score by this margin to displace it.
    churn_margin: float = Field(default=0.15, ge=0)
    #: ... and must hold that advantage for this many consecutive cycles.
    churn_persistence_cycles: int = Field(default=2, ge=1)
    scan_codes: list[str] = Field(
        default_factory=lambda: [
            "TOP_PERC_GAIN",
            "TOP_PERC_LOSE",
            "MOST_ACTIVE",
            "HOT_BY_VOLUME",
        ]
    )
    #: IBKR caps scanner results at 50 per scan code.
    scan_rows_per_code: int = Field(default=50, ge=1, le=50)
    scanner_min_price: float = 3.0
    scanner_min_volume: int = 500_000
    #: Cap on how many symbols may be promoted into the focus list per cycle.
    #: Each promotion costs one historical request, so this bounds intraday
    #: historical usage well under the 60-per-10-minutes ceiling.
    max_promotions_per_cycle: int = Field(default=4, ge=1, le=20)

    @model_validator(mode="after")
    def _clamp(self) -> UniverseConfig:
        self.scan_rows_per_code = min(self.scan_rows_per_code, HL.ABS_MAX_SCANNER_ROWS)
        if len(self.scan_codes) > HL.ABS_MAX_ACTIVE_SCANNERS:
            self.scan_codes = self.scan_codes[: HL.ABS_MAX_ACTIVE_SCANNERS]
        return self


class CadenceConfig(BaseModel):
    decision_interval_seconds: int = Field(default=300, ge=60, le=3600)
    fast_loop_seconds: float = Field(default=5.0, gt=0)
    account_sync_seconds: float = Field(default=30.0, gt=0)
    scanner_refresh_seconds: float = Field(default=60.0, gt=0)
    connection_check_seconds: float = Field(default=10.0, gt=0)
    dashboard_push_seconds: float = Field(default=1.0, gt=0)
    #: Minutes before the open to start the pre-market backfill.
    premarket_start_minutes_before_open: int = Field(default=150, ge=10)
    opening_range_minutes: int = Field(default=15, ge=1, le=60)


class ModelSpec(BaseModel):
    """Which model plays which role, and how it is decoded."""

    model: str = "gpt-oss:120b"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    num_ctx: int = Field(default=32768, ge=2048)
    num_predict: int = Field(default=2048, ge=128)
    seed: int | None = 42
    timeout_seconds: float = Field(default=180.0, gt=0)
    think: bool | str | None = None


class LLMConfig(BaseModel):
    #: "ollama_cloud" or "ollama_local".
    provider: str = "ollama_cloud"
    cloud_host: str = "https://ollama.com"
    local_host: str = "http://localhost:11434"
    #: Free = 1, Pro = 3, Max = 10. Calls serialize through a semaphore of this size.
    max_concurrent_requests: int = Field(default=1, ge=1, le=16)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(default=2.0, gt=0)
    #: Keep the model resident between cycles ("-1" never unloads).
    keep_alive: str = "30m"
    cache_enabled: bool = True
    audit_enabled: bool = True
    #: Token budget for the rolling session narrative before it is compacted.
    narrative_max_chars: int = Field(default=6000, ge=500)

    strategist: ModelSpec = Field(default_factory=ModelSpec)
    trader: ModelSpec = Field(default_factory=ModelSpec)
    risk_officer: ModelSpec = Field(default_factory=lambda: ModelSpec(num_predict=768))
    reviewer: ModelSpec = Field(default_factory=ModelSpec)


class BrokerConfig(BaseModel):
    host: str = "127.0.0.1"
    paper_port: int = 4002
    live_port: int = 4001
    client_id: int = 11
    connect_timeout: float = Field(default=20.0, gt=0)
    reconnect_base_delay: float = Field(default=5.0, gt=0)
    reconnect_max_delay: float = Field(default=300.0, gt=0)
    #: Sustained historical-data rate. IBKR allows 60 per 10 min; 10.5s is a safe period.
    historical_request_period: float = Field(default=10.5, gt=0)
    historical_burst: int = Field(default=6, ge=1)
    #: IBKR default is 100 concurrent market data lines.
    max_market_data_lines: int = Field(default=90, ge=1)
    #: Cancel an unfilled entry limit order after this long.
    order_timeout_seconds: float = Field(default=90.0, gt=0)
    #: Marketable offset applied to entry limit orders, as a fraction of price.
    limit_offset_pct: float = Field(default=0.001, ge=0, le=0.05)
    #: What to do with broker orders we do not recognize on reconnect.
    adopt_unknown_orders: bool = True
    #: Global outbound message cap. IBKR closes the socket above ~50/sec.
    max_messages_per_second: int = Field(default=45, ge=1)

    @model_validator(mode="after")
    def _clamp(self) -> BrokerConfig:
        self.max_market_data_lines = min(
            self.max_market_data_lines, HL.ABS_MAX_MARKET_DATA_LINES
        )
        self.max_messages_per_second = min(
            self.max_messages_per_second, HL.ABS_MAX_BROKER_MESSAGES_PER_SECOND
        )
        if self.paper_port == self.live_port:
            raise ValueError("paper_port and live_port must differ")
        return self


class DataConfig(BaseModel):
    directory: str = "data"
    duckdb_file: str = "data/bars.duckdb"
    backfill_days: int = Field(default=20, ge=2)
    bar_size: str = "5 mins"
    #: Bars kept in memory per symbol.
    memory_bars: int = Field(default=500, ge=50)


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    enabled: bool = True


class StrategyConfig(BaseModel):
    """The whole YAML file."""

    risk: RiskConfig = Field(default_factory=RiskConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    cadence: CadenceConfig = Field(default_factory=CadenceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> StrategyConfig:
        p = Path(path) if path else DEFAULT_CONFIG
        if not p.exists():
            return cls()
        raw: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
        return cls.model_validate(raw)


# --------------------------------------------------------------------------- #
# Environment-backed secrets and mode
# --------------------------------------------------------------------------- #


class Secrets(BaseSettings):
    """Everything that must never be committed."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    trading_mode: TradingMode = TradingMode.PAPER
    #: The second of three live-trading locks.
    allow_live_trading: bool = False
    #: The third: must match the account IBKR actually reports.
    ib_account_id: str = ""

    ollama_api_key: str = ""
    ollama_host: str = ""

    tws_userid: str = ""
    tws_password: str = ""

    log_level: str = "INFO"
    config_file: str = ""


class Settings(BaseModel):
    """Everything, assembled."""

    secrets: Secrets
    strategy: StrategyConfig
    repo_root: Path = Field(default_factory=Path.cwd)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Settings:
        secrets = Secrets()
        path = config_path or secrets.config_file or None
        return cls(secrets=secrets, strategy=StrategyConfig.load(path))

    # -- mode ------------------------------------------------------------- #

    @property
    def is_live(self) -> bool:
        return self.secrets.trading_mode == TradingMode.LIVE

    @property
    def port(self) -> int:
        b = self.strategy.broker
        return b.live_port if self.is_live else b.paper_port

    def verify_live_interlock(self, connected_account_id: str) -> None:
        """Assert the three live-trading locks agree. Called right after connect.

        Raises `LiveTradingNotPermitted` on any disagreement. This runs before a
        single order can be placed, and the process exits rather than continuing
        in an ambiguous state.
        """
        acct = (connected_account_id or "").upper()
        is_paper_account = acct.startswith(("DU", "DF"))

        if not self.is_live:
            # Paper mode: refuse to run against what looks like a live account.
            if acct and not is_paper_account:
                raise LiveTradingNotPermitted(
                    f"TRADING_MODE=paper but connected account {acct!r} is not a paper "
                    "account (paper accounts start with DU/DF). Refusing to start."
                )
            return

        if not self.secrets.allow_live_trading:
            raise LiveTradingNotPermitted(
                "TRADING_MODE=live but ALLOW_LIVE_TRADING is not true. "
                "Both must be set to trade real money."
            )
        if not self.secrets.ib_account_id:
            raise LiveTradingNotPermitted(
                "TRADING_MODE=live requires IB_ACCOUNT_ID to be set explicitly."
            )
        if acct != self.secrets.ib_account_id.upper():
            raise LiveTradingNotPermitted(
                f"Connected account {acct!r} does not match configured "
                f"IB_ACCOUNT_ID {self.secrets.ib_account_id!r}."
            )
        if is_paper_account:
            raise LiveTradingNotPermitted(
                f"TRADING_MODE=live but account {acct!r} is a paper account."
            )

    # -- paths ------------------------------------------------------------ #

    @property
    def data_dir(self) -> Path:
        d = self.repo_root / self.strategy.data.directory
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def kill_file(self) -> Path:
        return self.data_dir / "KILL"

    @property
    def duckdb_path(self) -> Path:
        return self.repo_root / self.strategy.data.duckdb_file

    @property
    def audit_dir(self) -> Path:
        d = self.data_dir / "audit"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def state_file(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def universe_file(self) -> Path:
        return self.repo_root / self.strategy.universe.symbols_file

    def load_universe(self) -> list[str]:
        """Read the broad symbol list. One symbol per line; '#' comments allowed."""
        p = self.universe_file
        if not p.exists():
            return []
        out: list[str] = []
        for line in p.read_text().splitlines():
            s = line.split("#", 1)[0].split(",", 1)[0].strip().upper()
            if s and s not in out:
                out.append(s)
        return out

    # -- llm -------------------------------------------------------------- #

    @property
    def ollama_host(self) -> str:
        if self.secrets.ollama_host:
            return self.secrets.ollama_host
        cfg = self.strategy.llm
        return cfg.cloud_host if cfg.provider == "ollama_cloud" else cfg.local_host

    @property
    def uses_cloud_llm(self) -> bool:
        return self.strategy.llm.provider == "ollama_cloud"


def get_settings(config_path: str | Path | None = None) -> Settings:
    return Settings.load(config_path or os.environ.get("AITRADER_CONFIG"))
