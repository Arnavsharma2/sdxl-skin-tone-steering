# Paper builds

The generic manuscript and author-identified preprint are maintained
separately. Build them with Tectonic:

```bash
make paper
make paper-arxiv
```

Run `make paper-audit` before release. The audit checks the central numerical
claims in both manuscripts against the retained evidence files.
