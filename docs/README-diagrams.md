# Diagrams

Mermaid sources live in `docs/*.mmd`.

To render PNG/PDF locally:

1. Install Node (LTS): https://nodejs.org/
2. Install Mermaid CLI:
```bash
npm i -g @mermaid-js/mermaid-cli
```
3. Run the renderer:
```bash
python scripts/render_diagrams.py
```

Outputs:
- `docs/architecture.png`, `docs/architecture.pdf`
- `docs/search-sequence.png`
- `docs/outfit-sequence.png`

If Mermaid CLI is not installed, the script prints instructions and exits non-zero.
