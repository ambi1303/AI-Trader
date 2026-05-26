"""Pydantic v2 contracts shared across layers.

Every layer reads/writes typed contracts, never bare dicts. This dramatically
reduces integration bugs and gives us a single source of truth for shapes.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BarSource(str, Enum):
    """Where an OHLCV bar came from. Used for cross-source validation."""

    YFINANCE = "yfinance"
    BHAVCOPY = "bhavcopy"
    NSEPYTHON = "nsepython"
    ANGELONE = "angelone"   # SmartAPI getCandleData (read-only, Week 5)


class CorporateActionType(str, Enum):
    SPLIT = "split"
    BONUS = "bonus"
    DIVIDEND = "dividend"
    RIGHTS = "rights"
    MERGER = "merger"
    DEMERGER = "demerger"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Symbol allow-list pattern (NSE):
# Uppercase letters, digits, hyphen and ampersand. 1-25 chars.
# Validated explicitly to avoid log/SQL injection via odd characters.
Symbol = Annotated[str, Field(pattern=r"^[A-Z0-9\-&]{1,25}$")]


class Bar(BaseModel):
    """A single daily OHLCV bar from a single source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: Symbol
    bar_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int = Field(ge=0)
    adj_close: Decimal | None = None
    source: BarSource
    ingested_at: datetime | None = None

    @field_validator("open", "high", "low", "close", "adj_close")
    @classmethod
    def _positive_price(cls, v: Decimal | None) -> Decimal | None:
        if v is None:
            return v
        if v <= 0:
            raise ValueError("price must be > 0")
        return v


class CorporateAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: Symbol
    ex_date: date
    action_type: CorporateActionType
    ratio_from: int | None = None
    ratio_to: int | None = None
    amount: Decimal | None = None
    notes: str | None = None
    source: str = "seed"


class ConstituencyEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: Symbol
    start_date: date
    end_date: date | None = None
    index_name: str = "NIFTY50"
    notes: str | None = None


class ValidationIssue(BaseModel):
    """Emitted by validators; persisted to validation_failures table."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    check_name: str
    symbol: Symbol | None = None
    issue_date: date | None = None
    severity: Severity
    message: str
    details: dict[str, str | int | float | None] | None = None
