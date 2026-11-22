"""Compatibility entry point for the current research evaluation runner."""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluation.run_revision_eval import main


if __name__ == "__main__":
    main()
