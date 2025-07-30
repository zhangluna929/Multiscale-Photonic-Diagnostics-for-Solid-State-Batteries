"""utils"""
import json, csv, pathlib
from typing import Any, Dict

__all__ = ["export_results"]

def export_results(obj: Dict[str, Any], file_path: str):
    """Dump dict to"""
    path = pathlib.Path(file_path)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(obj, indent=2))
    elif path.suffix.lower() == ".csv":
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            for k, v in obj.items():
                writer.writerow([k, v])
    else:
        raise ValueError("Unsupported file extension") 