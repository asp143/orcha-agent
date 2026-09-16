"""Generate bundled metadata: python scripts/generate_model_catalog.py /path/to/omp."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def generate(root: Path) -> dict[str, object]:
    source = json.loads((root / "packages/catalog/src/models.json").read_text())
    catalog = {}
    for provider, upstream in [
        ("anthropic", "anthropic"),
        ("openai", "openai"),
        ("codex", "openai-codex"),
        ("google", "google"),
        ("ollama", "ollama"),
    ]:
        for name, model in source.get(upstream, {}).items():
            catalog[f"{provider}:{name}"] = {
                "provider": provider,
                "id": name,
                "context_window": model["contextWindow"],
                "max_tokens": model["maxTokens"],
                "cost": {
                    key: model.get("cost", {}).get(original, 0)
                    for key, original in [
                        ("input", "input"),
                        ("output", "output"),
                        ("cache_read", "cacheRead"),
                        ("cache_write", "cacheWrite"),
                    ]
                },
                "thinking": bool(model.get("reasoning")),
                "vision": "image" in model.get("input", []),
                "tool_calls": model.get("compat", {}).get("supportsToolChoice", True),
            }
    return catalog


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "orcha_agent/core/catalog/models.json"
    target.write_text(json.dumps(generate(Path(sys.argv[1])), indent=2) + "\n")
