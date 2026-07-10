"""Smoke test: verify the dataset registry loads from YAML config."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cbt_companion.core import DatasetRegistry
from cbt_companion.models import SourceType

config_path = Path(__file__).resolve().parent.parent / "configs" / "datasets.yaml"
registry = DatasetRegistry.from_yaml(config_path)

print(repr(registry))
print(f"Entries: {len(registry)}")
print()

for entry in registry:
    print(f"  key         : {entry.key}")
    print(f"  name        : {entry.name}")
    print(f"  source      : {entry.source.value}")
    print(f"  dataset_id  : {entry.dataset_id}")
    print(f"  local_path  : {entry.local_path}")
    print(f"  description : {entry.description[:70]}...")
    print()

# ── Assertions ────────────────────────────────────────────────

# Registry length
assert len(registry) == 3, f"Expected 3 entries, got {len(registry)}"

# Lookup by key
assert registry["esconv"].name == "ESConv"
assert registry["empathetic_dialogues"].dataset_id == "facebook/empathetic_dialogues"
assert registry["mental_health_counseling_conversations"].dataset_id == "Amod/mental_health_counseling_conversations"

# Source type
assert registry["esconv"].source is SourceType.HUGGINGFACE
assert registry["empathetic_dialogues"].source is SourceType.HUGGINGFACE
assert registry["mental_health_counseling_conversations"].source is SourceType.HUGGINGFACE

# Local paths
assert registry["esconv"].local_path == Path("datasets/raw/esconv")

# Containment
assert "esconv" in registry
assert "nonexistent" not in registry

# .get() returns None for missing keys
assert registry.get("nonexistent") is None

# .keys()
assert set(registry.keys()) == {"empathetic_dialogues", "esconv", "mental_health_counseling_conversations"}

# .list_all()
assert len(registry.list_all()) == 3

# Iteration
assert len(list(registry)) == 3

# Frozen model — should not allow mutation
try:
    registry["esconv"].name = "changed"  # type: ignore[misc]
    assert False, "DatasetConfig should be frozen"
except Exception:
    pass  # Expected: Pydantic frozen model raises on mutation

print("All assertions passed.")
