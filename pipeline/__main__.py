"""Allow running the pipeline package as ``python -m pipeline``."""

from __future__ import annotations

import sys

from pipeline.cli import main

sys.exit(main())
