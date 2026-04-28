"""DPO ablation: congestion weight 1.25."""
import importlib.util
from pathlib import Path

_base_path = Path(__file__).parent / "ablation_base.py"
_spec = importlib.util.spec_from_file_location("ablation_base", str(_base_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AblationDPOPlacer = _mod.AblationDPOPlacer

class DPOCW125Placer(AblationDPOPlacer):
    def __init__(self):
        super().__init__(congestion_weight=1.25)
