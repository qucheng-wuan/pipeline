"""Checkout-local entrypoint for pinned official and private release assets."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from transparent_depth.assets import main

if __name__ == '__main__':
    main()
