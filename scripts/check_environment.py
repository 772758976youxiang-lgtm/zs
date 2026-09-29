import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.system import system_info

print(json.dumps(system_info(), ensure_ascii=False, indent=2))
