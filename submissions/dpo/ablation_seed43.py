"""DPO multi-seed: seed 43."""
import importlib.util
from pathlib import Path

_base_path = Path(__file__).parent / "placer.py"
_spec = importlib.util.spec_from_file_location("dpo_placer", str(_base_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DPOPlacer = _mod.DPOPlacer

class DPOSeed43Placer(DPOPlacer):
    def __init__(self):
        super().__init__(seed=43)
