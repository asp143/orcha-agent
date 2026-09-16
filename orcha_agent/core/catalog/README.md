Bundled metadata is generated from oh-my-pi `packages/catalog/src/models.json`
at revision `042028fd018b1282fbe660ab7255ebad18dd4db5`.

Regenerate from a read-only checkout:

```sh
python scripts/generate_model_catalog.py /path/to/omp
```

The conversion retains anthropic, openai, google and ollama providers, and maps
upstream openai-codex to orcha's codex prefix. The referenced snapshot contains
no ollama rows: local model metadata is supplied by models.yml. The langchain
provider is a passthrough with unknown context/cost unless overridden. Context
and output limits are tokens; all costs are USD per million tokens. Reasoning
becomes thinking, image input becomes vision, and tool-choice support supplies
the tool_calls flag (true when the source omits that compatibility field).
