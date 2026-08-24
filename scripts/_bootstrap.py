"""Đưa `src/` vào sys.path. Import module này ĐẦU TIÊN trong mọi script.

Repo không đóng gói thành package pip (xem pyproject.toml), nên script cần tự trỏ
đường tới source root.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
