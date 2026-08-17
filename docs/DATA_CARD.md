# Data card: synthetic paired portrait campaigns

## Dataset purpose

The dataset estimates and validates a visual skin-tone direction for controlled
SDXL research. It is entirely synthetic and contains no intentionally collected
photographs of real people. Synthetic faces can nevertheless resemble real
people by chance and should be handled as potentially sensitive portrait data.

## Confirmatory campaign

- Generator: SDXL 1.0 base, immutable revision
  `462165984030d82259a11f4367a4eed129e94a7b`
- Size: 96 paired prompts / 192 images at 1024x1024
- Direction split: pairs 0--63
- Held-out measurement split: pairs 64--95
- Pairing: identical initial diffusion seed and fixed prompt template, with
  only the skin-tone descriptor changed
- Manifest SHA-256:
  `382a77c27aa6763e83133bb2f89fb249ad02f692e1775c7ef98c9eebf520a24d`
- Generation ledger: one content-addressed attestation per image

The descriptors cycle over eight light-complexion phrases and eight
dark-complexion phrases. Those labels are prompt instructions, not verified
demographic identities. Prompt wording can change age, gender presentation,
hair, facial structure, lighting, and composition despite noise coupling.

## Independent replication campaign

The prospective replication uses the same generator, descriptors, and split
sizes but a new frozen paired-data seed schedule. Its planned config SHA-256 is
`c6121300529a1d218b0891f953dd7d787043feb44aae870b7c0da2722fcde2b3`.
Its 30 evaluation seeds are also disjoint from the original study. All 192
images were generated and content-attested. The manifest SHA-256 is
`71dfb6ee8e8a8ef2381a4521b0d9110a7a5b1f97d515d76413c3e0f8b5fbe12c`,
the held-out validation report SHA-256 is
`c47967baf10fec5f361c4239c95999ee9a4a293cf8bfcff85141b4b0b2477541`,
and the resulting immutable replication config fingerprint is
`fa694f8bb214c219`.
The complete content-attested image archive SHA-256 is
`f638e8be419dfc542b9d4a7b295675ceb9b7bdf0db8ce7dca4c689ff519966c5`.

## Collection and quality controls

- Exact model revision, inference steps, guidance, image size, prompts, and
  negative prompt are fingerprinted.
- Existing files are reused only when their path, descriptor, seed, generation
  signature, and SHA-256 match the append-only campaign ledger.
- A manifest is considered complete only when both images of all 96 pairs have
  valid attestations.
- Held-out measurement validation requires at least 95% detection, at least
  90% pair ordering, a median separation of at least eight ITA degrees, and
  bounded response to four prespecified illumination perturbations.

## Sensitive attributes and privacy

The study intentionally varies apparent skin colour and evaluates facial
structure with pretrained face models. No names, source identities, real-person
labels, or face embeddings are released. Only aggregate similarity values are
stored in result ledgers. The generated images should not be linked to real
identities or used to train profiling systems.

## Distribution

Manifests, hashes, configs, aggregate results, and a minimal set of paper
examples are suitable for public release after project-license approval. Full
image campaigns remain external, optional research artifacts with an SDXL
output-use notice. SDXL and evaluation weights are never included.

## Known biases and limitations

- The same generator produces the direction data and evaluation images.
- Binary light/dark descriptors do not represent the continuous diversity of
  skin appearance and do not correspond to race or ethnicity.
- Controlled studio prompts underrepresent lighting, pose, age, presentation,
  and environmental diversity.
- Detector and embedding failures may vary with apparent skin tone.
- Model-generated portraits can reproduce stereotypes and correlations from
  SDXL training data.
