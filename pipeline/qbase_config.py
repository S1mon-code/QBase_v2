"""Centralized QBase_v2 configuration.

Loads config/settings.yaml and provides constants used across all modules.
Replaces hardcoded paths scattered across pipeline/, optimizer/, scripts/.
"""
from __future__ import annotations

import sys
from pathlib import Path
from functools import lru_cache

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@lru_cache(maxsize=1)
def _load_settings() -> dict:
    path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

def get_settings() -> dict:
    return _load_settings()

# ── Paths ─────────────────────────────────────────────────
ALPHAFORGE_PATH = str(Path("/Users/simon/Desktop/AlphaForge"))
DATA_DIR = str(Path(ALPHAFORGE_PATH) / "data")

# Ensure both are on sys.path
for _p in (ALPHAFORGE_PATH, str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Capital & Risk ────────────────────────────────────────
def _s():
    return get_settings()

INITIAL_CAPITAL = 10_000_000
RISK_PER_TRADE = 0.02
MAX_MARGIN_SINGLE = 0.30
MAX_MARGIN_TOTAL = 0.80

# ── Vol Targeting ─────────────────────────────────────────
TARGET_VOL = 0.15  # Used by backtest_runner / signal_blender
VOL_LOOKBACK = 20

# ── Backtest ──────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.03

# ── Instruments ───────────────────────────────────────────
INSTRUMENTS = ["RB", "HC", "I", "J", "JM"]
TIMEFRAMES = ["daily", "1h", "2h", "4h"]

# ── Direction mapping ─────────────────────────────────────
DIR_MAP = {"long": "up", "short": "down"}
