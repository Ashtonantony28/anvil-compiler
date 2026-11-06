import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The simulator's Python packages (xf, test.helpers) are imported from the pinned submodule.
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT))
