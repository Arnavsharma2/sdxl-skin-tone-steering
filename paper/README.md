# Paper builds

The generic manuscript, author-identified arXiv preprint, and anonymous WACV
submission are maintained separately. Build them with Tectonic:

```bash
make paper
make paper-arxiv
make paper-wacv
```

Run `make paper-audit` before release to verify the manuscript's central
numerical claims against retained evidence. Do not use the author-identified
arXiv source for double-blind conference submission.
