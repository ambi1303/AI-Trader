"""Download daily NSE BhavCopy and persist as Bar contracts.

NSE switched the BhavCopy format on 2024-07-08. Both formats are handled.

- Old (pre 2024-07-08):
    URL pattern:  archives.nseindia.com/.../cm{DD}{MMM}{YYYY}bhav.csv.zip
    Columns:      SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, LAST, PREVCLOSE,
                  TOTTRDQTY, TOTTRDVAL, TIMESTAMP, TOTALTRADES, ISIN
- New (>= 2024-07-08):
    URL pattern:  nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip
    Columns vary; we use the common set: TckrSymb, SctySrs, OpnPric,
                  HghPric, LwPric, ClsPric, LastPric, TtlTradgVol, TradDt
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import yaml

from src.contracts import Bar, BarSource
from src.data_ingestion.nse_session import make_session, warm_up, get_bytes
from src.utils.logger import get_logger
from src.utils.secrets import project_root

log = get_logger("ingest.bhavcopy")

_CONFIG_PATH = project_root() / "config" / "config.yaml"


@dataclass(frozen=True)
class BhavcopyConfig:
    cutover_date: date
    old_url_template: str
    new_url_template: str


def _load_config() -> BhavcopyConfig:
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = raw["ingestion"]["bhavcopy"]
    return BhavcopyConfig(
        cutover_date=datetime.fromisoformat(cfg["cutover_date"]).date(),
        old_url_template=cfg["old_url_template"],
        new_url_template=cfg["new_url_template"],
    )


def _build_url(d: date, cfg: BhavcopyConfig) -> str:
    if d < cfg.cutover_date:
        return cfg.old_url_template.format(
            yyyy=f"{d.year:04d}",
            mmm_upper=d.strftime("%b").upper(),
            dd=f"{d.day:02d}",
        )
    return cfg.new_url_template.format(yyyymmdd=d.strftime("%Y%m%d"))


def _parse_old(zip_bytes: bytes, d: date) -> list[Bar]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as f:
            df = pd.read_csv(f)
    df.columns = [c.strip().upper() for c in df.columns]
    eq = df[df["SERIES"].str.strip() == "EQ"].copy()
    bars: list[Bar] = []
    for _, row in eq.iterrows():
        try:
            bars.append(
                Bar(
                    symbol=str(row["SYMBOL"]).strip().upper(),
                    bar_date=d,
                    open=Decimal(str(row["OPEN"])),
                    high=Decimal(str(row["HIGH"])),
                    low=Decimal(str(row["LOW"])),
                    close=Decimal(str(row["CLOSE"])),
                    volume=int(row["TOTTRDQTY"]),
                    adj_close=None,  # raw bhavcopy is not adjusted
                    source=BarSource.BHAVCOPY,
                )
            )
        except Exception as e:  # row-level skip with audit log
            log.warning("Skipping row in old bhavcopy {}: {}", d, str(e))
    return bars


def _parse_new(zip_bytes: bytes, d: date) -> list[Bar]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as f:
            df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    # Series column name is SctySrs in new format
    series_col = "SctySrs" if "SctySrs" in df.columns else "Sctysrs"
    eq = df[df[series_col].astype(str).str.strip() == "EQ"].copy()
    bars: list[Bar] = []
    for _, row in eq.iterrows():
        try:
            bars.append(
                Bar(
                    symbol=str(row["TckrSymb"]).strip().upper(),
                    bar_date=d,
                    open=Decimal(str(row["OpnPric"])),
                    high=Decimal(str(row["HghPric"])),
                    low=Decimal(str(row["LwPric"])),
                    close=Decimal(str(row["ClsPric"])),
                    volume=int(row["TtlTradgVol"]),
                    adj_close=None,
                    source=BarSource.BHAVCOPY,
                )
            )
        except Exception as e:
            log.warning("Skipping row in new bhavcopy {}: {}", d, str(e))
    return bars


def fetch_bhavcopy(d: date, *, cache_dir: Path | None = None) -> list[Bar]:
    """Fetch BhavCopy for a single date. Returns parsed Bar list (EQ series only)."""
    cfg = _load_config()
    url = _build_url(d, cfg)
    cache_dir = cache_dir or (project_root() / "data" / "raw" / "bhavcopy")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"bhavcopy_{d.isoformat()}.zip"

    if cache_file.exists():
        zip_bytes = cache_file.read_bytes()
        log.debug("Cache hit: {}", cache_file.name)
    else:
        sess = make_session()
        warm_up(sess)
        log.info("Fetching BhavCopy for {} from {}", d.isoformat(), url)
        try:
            zip_bytes = get_bytes(sess, url)
        except Exception as e:
            log.error("BhavCopy fetch failed for {}: {}", d.isoformat(), str(e))
            return []
        cache_file.write_bytes(zip_bytes)

    if d < cfg.cutover_date:
        return _parse_old(zip_bytes, d)
    return _parse_new(zip_bytes, d)
