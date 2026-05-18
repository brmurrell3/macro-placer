"""K_eps=3 variant of stacked_periphery."""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
spec = importlib.util.spec_from_file_location('s', str(_ROOT / 'submissions' / 'cd_lns_sa_cascade_stacked_periphery' / 'placer.py'))
_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(_mod)

class CDLNSSACascadeStackedPeripheryK3Placer(_mod.CDLNSSACascadeStackedPeripheryPlacer):
    """K_eps=3 (vs default 2) for more Lévy magnitude diversity."""
    def __init__(self, **kwargs):
        kwargs.setdefault('portfolio_K_eps', 3)
        super().__init__(**kwargs)
