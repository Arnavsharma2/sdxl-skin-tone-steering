# Contributing

Contributions that improve correctness, reproducibility, documentation, or
responsible-use safeguards are welcome.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[evaluation,dev]'
make check
make paper-audit
```

## Pull requests

- Keep changes focused and explain their effect on retained evidence.
- Add or update tests for changes to study logic, metrics, or audits.
- Run `make check` and `make paper-audit` before opening a pull request.
- Do not strengthen manuscript claims without updating the claims-to-evidence
  register and providing retained supporting evidence.
- Do not commit model weights, generated portrait archives, real-person images,
  face embeddings, credentials, local paths, or private review materials.
- Preserve failure and missingness records; do not silently filter them from
  analyses.

By contributing, you agree that your contribution is licensed under the
repository's Apache-2.0 license.
