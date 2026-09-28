import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ops_agents import config  # noqa: E402

if not config.DB_PATH.exists():
    from build_database import build  # noqa: E402

    build(config.DB_PATH)
