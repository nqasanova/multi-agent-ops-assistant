"""Paths and settings, read once from the environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DOCS_DIR = ROOT / "data" / "docs"
DB_PATH = Path(os.getenv("OPS_DB_PATH", ROOT / "data" / "ops.db"))

# Retrieval
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# SQL guardrails
SQL_ROW_LIMIT = int(os.getenv("SQL_ROW_LIMIT", "50"))

# Agent loop limits
MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "6"))
MAX_WRITE_ATTEMPTS = int(os.getenv("MAX_WRITE_ATTEMPTS", "2"))
