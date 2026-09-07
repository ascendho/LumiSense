from __future__ import annotations

from typing import Any


def flatten_numeric(snapshot: dict[str, Any], prefix: str = "") -> list[tuple[str, float]]:
    output = []
    for key, value in snapshot.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            output.extend(flatten_numeric(value, f"{path}."))
        elif isinstance(value, bool):
            output.append((path, float(int(value))))
        elif isinstance(value, (int, float)):
            output.append((path, float(value)))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    output.extend(flatten_numeric(item, f"{path}."))
    return output


def extract_numeric_features(snapshot: dict[str, Any]) -> dict[str, float]:
    features = {}
    for path, value in flatten_numeric(snapshot):
        key = path.replace("device_self_report.", "").replace("gateway_observation.", "")
        features.setdefault(key, value)
    return features


def event_text(snapshot: dict[str, Any]) -> str:
    return " ".join(
        event.get("message", "")
        for event in snapshot.get("events", [])
        if event.get("message")
    )
