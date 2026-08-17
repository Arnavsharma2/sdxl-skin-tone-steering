# Artifact availability and reproducibility boundary

Status: prepared for anonymous review on 2026-08-17.

## Available in the Git checkout and anonymous supplement

The repository and supplementary ZIP contain the exact source, environment
lock, frozen configs, paired-data generation ledgers and manifests, complete
570-row result ledgers for both campaigns, aggregate analysis tables, figures,
tests, and the 16-group manuscript-claim audit. From these files a reviewer can:

- reproduce the aggregate analyses and locked replication decision;
- verify every central numerical manuscript claim against retained evidence;
- rebuild the deterministic anonymous supplementary ZIP; and
- inspect the generation and evaluation implementation.

The tracked paired-data manifests attest filenames, generation signatures, and
content hashes. They do not contain the paired PNGs themselves.

## Retained separately

The following large or sensitive artifacts are not in Git and are not included
in the under-200-MiB WACV supplement:

| Artifact | Size | SHA-256 | Needed for |
|---|---:|---|---|
| Parent frozen-input archive | 212 MiB | `0291ad0335e90739ecd3caeb65b13d5a252581d6de99db021aca791eaacf9267` | Rechecking the 192 paired direction/validation images and pre-outcome freeze bundle. |
| Parent full-run archive | 678 MiB | `66973e4ce3fa5a43655b694738f60efeee983dd46bdaec4961b9dbf8ff924da4` | Recomputing parent image metrics and the full run-integrity audit. |
| Replication frozen-input archive | 52 KiB | `fb352010ea020f43446e30cf4c251506a844bdcb87a3fdfdb21a6553ee931184` | Inspecting the replication freeze bundle and paired-data ledger. |
| Replication slim archive | 628 KiB | `19e309fee99462cf3cdc6a164c120be9bbe5a000c02040719db499fd81a26d07` | Reproducing the locked replication analysis with direction tensors. |
| Replication full-run archive | 663 MiB | `be587ca138878736c9bad55758701e6d66c4d8d63772361810c6eb5023a42fbd` | Recomputing replication image metrics and the full run-integrity audit. |
| Replication paired-image archive | 213 MiB | `f638e8be419dfc542b9d4a7b295675ceb9b7bdf0db8ce7dca4c689ff519966c5` | Rechecking the 192 independent direction/validation images. |

These archives deliberately exclude third-party model weights and face
embeddings. The full image archives contain synthetic portraits that can
resemble real people and therefore require a separate release and misuse
decision.

## Submission disclosure

Until an anonymous, stable reviewer-accessible URL is added, the paper and
supplement must describe the large archives as retained but unavailable through
the review package. A checksum is an integrity identifier, not proof of public
availability or of pre-outcome timing.

Before submission, the corresponding author must choose one of two honest
options:

1. provide a stable anonymous link to the retained archives, preserving the
   hashes above and WACV anonymity; or
2. leave the archives unavailable during review and retain the explicit
   limitation that image-level metric recomputation and full run-integrity
   verification require author-held artifacts.

After acceptance, publish the releasable subset under a versioned DOI or release
asset, with the repository license, output-use notice, and archive manifest.
