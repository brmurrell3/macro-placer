"""DPO ablation: congestion weight 0.50 (baseline control)."""
import importlib.util
from pathlib import Path

_base_path = Path(__file__).parent / "ablation_base.py"
_spec = importlib.util.spec_from_file_location("ablation_base", str(_base_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AblationDPOPlacer = _mod.AblationDPOPlacer

class DPOCW050Placer(AblationDPOPlacer):
    def __init__(self):
        super().__init__(congestion_weight=0.50)
