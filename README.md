```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# Sync dependencies
uv sync

# Run Marimo
uv run marimo edit main.py
```

if you need to install dependencies:

```bash
uv add 'NAME'

```

for help with AI use ```CLAUDE.md``` it has rules that would help write  correct code
