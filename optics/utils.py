"""utils.py
I/O helpers.
"""
import json, csv, pathlib
from typing import Any, Dict

__all__ = ["export_results"]

def export_results(obj: Dict[str, Any], file_path: str):
    """Dump dict to .json or .csv depending on extension."""
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

__all__.append("export_results_kafka")


def export_results_kafka(topic: str, msg: dict, broker: str = "localhost:9092"):
    """Send dict as JSON string to Kafka; fallback to print if lib/broker absent."""
    import json, time
    try:
        from confluent_kafka import Producer  # type: ignore
        p = Producer({"bootstrap.servers": broker})
        payload = json.dumps(msg).encode()
        p.produce(topic, payload)
        p.flush()
    except Exception as e:
        print(f"[warn] Kafka send failed: {e}. Payload logged locally.")
        ts = int(time.time())
        pathlib.Path(f"kafka_backup_{ts}.json").write_text(json.dumps(msg, indent=2)) 