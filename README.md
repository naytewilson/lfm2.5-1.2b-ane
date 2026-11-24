# LFM2.5-1.2B-Instruct on the Apple Neural Engine

A 4-bit Core ML build of Liquid AI's LFM2.5-1.2B-Instruct that runs every layer on the
Apple Neural Engine (ANE) of an A18 Pro and decodes at **58.1 tokens per second**
(median of five runs), with the qualification evidence behind each number on this page.

**Status: measured, v0.1.0 evidence release.** Placement, decode speed across three
context lengths, ANE power, a resident local HTTP endpoint and agreement with the source
model are measured on one machine. The Core ML packages and the runner are not attached
yet; see [Release assets](#release-assets). Read [Limitations](#limitations) before
relying on any of it.

Not affiliated with or endorsed by Liquid AI. LFM2.5 is Liquid AI's model; derivative
files are distributed under the [LFM Open License v1.0](LICENSE). See [NOTICE.md](NOTICE.md).

## Results

All figures: one Apple A18 Pro MacBook (`Mac17,5`, 8 GB, macOS 27.3 build `26C5047e`),
greedy decoding, one token per ANE call, measured 2026-11-23 to 2026-11-24.

| Measure | Result |
|---|---|
| Decode, 256-position context (5 runs of 49 tokens) | **58.1 tok/s median**; mean 58.56, 95% CI 52.1–65.0 (runs: 51.1, 58.1, 63.1, 63.8, 56.7) |
| Decode, 512 positions (2 runs) | 53.45 tok/s mean (56.4, 50.5) |
| Decode, 1024 positions (2 runs) | 51.45 tok/s mean (48.4, 54.5) |
| Benchmark verdict | 9 of 9 runs valid under all six harness gates |
| Operations planned on the ANE | **885 of 885** (100%); no CPU fallbacks for state |
| ANE power during decode (256 positions) | 1.99 W ANE rail, 3.21 W active package; **18.1 tok/s per package watt** (29.3 per ANE watt) |
| Memory traffic during decode | 38.7 GB/s DRAM; ANE DMA active 93.9% of the time |
| Resident HTTP server (OpenAI-style endpoint) | loads once: 4 requests with 0 reloads, repeated requests bit-identical, clean restart and shutdown, 59.7 MB peak memory |
| Weights on the device | 658 MB (weight-byte ratio 0.295 in the encoding audit) |

Context costs about 3 µs per position per token (fitted slope +3.02 µs/pos), so speed falls
slowly as the context grows.

### How it was measured

- **Decode speed.** Each run is a separate process that loads the model and generates 49
  tokens with stateful decode; its speed is 1000 divided by the median step time in
  milliseconds, so prompt prefill is excluded. A run counts only if it passes all six
  harness gates, including at least 20 s of wall time and 1 s of CPU time, fewer than
  50,000 major page faults and consistent repeats. The 95% interval is a Student-t
  interval over the five 256-position runs.
- **Placement.** Core ML's compute plan for the compiled model, read per operation.
  "On the ANE" means the plan assigns the operation to the Neural Engine; it is a
  plan-level statement, not a hardware execution trace. The token embedding lookup is a
  host-side gather in a separate segment; all 16 layers and the output head are in the
  ANE segment.
- **Power.** On-device rail telemetry, sampled during decode after an idle gate
  (0.77 W package, 0 W ANE at idle). Watts are attributed by rail, not per operation.
- **Encoding.** A static audit of the compiled program: 67 LUT4 and 26 LUT6 palettized
  weight operations, no affine or dense weight matrices.

### Agreement with the source model

| Check | Result |
|---|---|
| Compiler certificate (structure, encoding, bit-exactness, hardware) | 13 pass, 0 fail; max abs ANE−CPU difference 0.0566 |
| 32-bit reference against Liquid AI's greedy trajectory | 31 of 32 tokens; the one difference is a near-tie between 32-bit and bfloat16 |
| This build vs the 32-bit model, teacher-forced over 49 tokens | logit cosine 0.924, top-1 37 of 49, mean KL 0.856 |
| Same program on GPU (`cpuAndGPU`) vs ANE | cosine 0.9924, top-1 5 of 5, mean KL 0.0045 |
| Same program on CPU (`cpuOnly`) vs ANE | cosine 0.9901, top-1 5 of 5, mean KL 0.0059 |

## The build

| | |
|---|---|
| Source checkpoint | [`LiquidAI/LFM2.5-1.2B-Instruct`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) at revision `868df74dd56ff8a0c2ac5dbf281690c2dbebe4c9` |
| Architecture | 16 layers (10 gated short-convolution, 6 grouped-query attention with QK norm), hidden 2048, FFN 8192, vocabulary 65,536, tied output head |
| Quantization | 4-bit palettized weights, one lookup table per 4 rows; the `w2` and `in_proj` weights use 6-bit tables; range calibration |
| State | Convolution and attention caches kept as Core ML program state, with no CPU fallback |
| Contexts | Separate programs for 256, 512 and 1024 positions |

### A finding on palette group size (A18 Pro)

With placement held at 100% ANE in one segment, the size of the row group that shares a
lookup table changes decode speed sharply:

| Tables | Step time | Speed |
|---|---|---|
| 4-bit, one table per 8 rows | 14.11 ms | 70.9 tok/s |
| 4-bit, one table per 4 rows | 15.84 ms | 63.1 tok/s |
| 4-bit per 4 rows, 6-bit `w2` and `in_proj` (this build) | 17.22 ms | 58.1 tok/s |
| 4-bit, one table per row | 48.80 ms | 20.3 tok/s |

One table per row is 3.46 times slower than one per 4 rows at nearly the same weight
size. Finer tables are more accurate (teacher-forced top-1 39 of 49 per row, against 37 of
49 for this build), which is the trade this build makes between speed and agreement.

## Limitations

- **Accuracy is close to, not equal to, the source model.** Teacher-forced top-1 agreement
  is 37 of 49 tokens. The GPU and CPU paths agree with the ANE at cosine 0.992 and 0.990,
  below the 0.9999 bar this project uses for exact parity, and that comparison is between
  devices running this quantized program, not against the original weights.
- **One machine.** Every number comes from one A18 Pro MacBook on one OS build. Other
  chips, other macOS builds and thermal conditions are not measured. The A18 Pro is
  fanless; long runs can be slower.
- **No task benchmarks.** No standard evaluation suite (MMLU, GSM8K or similar) has been
  run on this build.
- **Short runs and short contexts.** Each speed run generates 49 tokens; long generations
  of several hundred tokens, where a fanless chip heats up, have not been measured for
  this build. Programs exist for up to 1024 positions only.
- **Placement is plan-level.** See How it was measured.

## Release assets

Not attached yet. The qualified build is defined by its compile flags and hashes in
[`evidence/qualification-20261123/qualification.json`](evidence/qualification-20261123/qualification.json).
The packages and runner will be attached to the v0.1.0 release only after a rebuild from
the public checkpoint reproduces those hashes and passes the same qualification.

## Evidence

[`evidence/qualification-20261123/`](evidence/qualification-20261123/) holds the
machine-readable receipts behind this page: the qualification summary, the benchmark
receipt and per-run results, the rail telemetry for each context, the encoding and
placement audit, the hardware check and the resident-server runs. Local file paths and
the host name were redacted before publication, so these files' hashes differ from the
lab originals; no measured value was changed.

## Related

- [lfm3-1.7b-ane](https://github.com/naytewilson/lfm3-1.7b-ane): LFM3-1.7B on the ANE
- [qwen2.5-ane](https://github.com/naytewilson/qwen2.5-ane): Qwen2.5-1.5B-Instruct on the ANE
- [macbrainslabs.org](https://macbrainslabs.org)

## License

Derivative files are distributed under the [LFM Open License v1.0](LICENSE), the license
of the source model. This README and everything under `evidence/` are original to this
repository.
