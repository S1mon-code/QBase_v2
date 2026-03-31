"""Ensure QBase_v2 root and AlphaForge are on sys.path."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# AlphaForge path — loaded from config at runtime
# See config/settings.yaml for alphaforge_path
