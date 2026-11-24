# Notice of changes

This repository and its release assets contain, or will contain, derivative works of
**LFM2.5-1.2B-Instruct** by Liquid AI, Inc. (`LiquidAI/LFM2.5-1.2B-Instruct`, revision
`868df74dd56ff8a0c2ac5dbf281690c2dbebe4c9`), distributed under the LFM Open License v1.0
(see `LICENSE`). They are not produced, reviewed or endorsed by Liquid AI.

Modifications made to the upstream work in the Core ML packages:

| Change |
|---|
| Weights palettized to 4-bit lookup tables (one table per 4 rows), with 6-bit tables for the `w2` and `in_proj` weights; range calibration applied |
| Weights and constants rounded to IEEE binary16 and converted to Core ML `mlprogram` format |
| Model split into a host segment (token embedding lookup) and an ANE segment (all layers and the output head), with caches held as program state |
| Separate programs compiled for 256, 512 and 1024 positions |

`LICENSE` is an unmodified copy of the upstream license file. This repository's
`README.md`, `NOTICE.md`, `CITATION.cff`, `SECURITY.md` and everything under `evidence/`
are original to this repository.
