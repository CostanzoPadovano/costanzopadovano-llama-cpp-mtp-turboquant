# Qwen3.6 35B A3B MTP OpenCode Checkpoint Notes

This document records a local fork experiment for OpenCode-style workflows on
Qwen3.6 35B A3B MTP GGUF with TurboQuant KV cache on Windows/CUDA Blackwell.

The goal was to avoid repeated full prompt re-prefill when OpenCode sends
large prompts that share long prefixes but diverge near the tail.

This is a local experimental result, not an upstream support claim.

## Test Setup

Hardware:

- Windows
- 2 x NVIDIA GeForce RTX 5060 Ti 16 GB
- CUDA Blackwell, compute capability 12.0

Model:

- `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`
- `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
- MTP enabled with `--spec-type mtp --spec-draft-n-max 2`

Primary OpenCode profile:

```text
ctx = 150000
batch = 2048
ubatch = 256
gpu_layers = 999
K cache = q8_0
V cache = turbo3
tensor_split = 0.95,1.05
reasoning_budget = 2048
cache_ram = 0
checkpoint_interval = 2048
checkpoint_max = 96
```

The local launcher is outside this repository:

```text
C:\MYPROJECT\Bioinformatics_Agent\scripts\launchers\mtp-turboquant\qwen36_35b_mtp_turboquant_max_bench_150k_q4_k_xl.bat
```

## Code Changes

The fork adds CLI aliases around the existing slot checkpoint settings:

```text
--checkpoint-interval N
--slot-checkpoint-interval N
--checkpoint-max N
--slot-checkpoint-max N
```

These map to the existing internals:

```text
--checkpoint-every-n-tokens
--ctx-checkpoints
```

The fork also accepts `--np` as an alias for `-np` / `--parallel`, and adds a
server helper flag:

```text
--debug-slot
```

`--debug-slot` sets `LLAMA_SERVER_SLOTS_DEBUG=1` before server slot logging is
initialized.

## Checkpoint Creation

Before this experiment, the default checkpoint cadence was too sparse for the
OpenCode workload. With the default interval, the first useful checkpoint could
be around 8192 tokens. If a later prompt had:

```text
n_past = 6798
first checkpoint = 8191
```

then the checkpoint was after the shared prefix and could not be restored. The
server had to reprocess from token 0.

With:

```text
--checkpoint-interval 2048
```

the same prompt can usually restore from:

```text
2048, 4096, 6144, ...
```

For `n_past = 6798`, the slot can restart from the checkpoint around `6144`
instead of 0.

The fork also creates early anchor checkpoints near important prefix sizes:

```text
1024, 2048, 4096, 6144, 8192, 12288, 16384,
24576, 32768, 49152, 65536
```

## Restore Policy

The restore path now selects a checkpoint only if it is actually before the
longest common prefix:

```text
checkpoint.n_tokens <= n_past
```

This prevents choosing a checkpoint that is structurally close in position but
past the reusable token prefix.

When a checkpoint is restored, the fork increments lightweight metadata:

```text
n_hits
t_last_used
```

Those fields are used by the eviction policy.

## Eviction Policy

The old behavior was effectively FIFO when the per-slot checkpoint limit was
reached. That is not ideal for OpenCode because the workflow often returns to
short or middle prompt prefixes after building a long tail.

The new eviction policy prefers to keep:

- early anchors
- recently created tail checkpoints
- checkpoints that have been restored before
- checkpoints that are not redundant with a nearby neighbor

It prefers to evict checkpoints that are close to other checkpoints, have no
hits, are not anchors, and are not in the recent tail window.

The practical memory tradeoff observed locally is still about 62.8 MiB per
checkpoint for this model/profile. That means:

```text
64 checkpoints  ~= 4.0 GiB
96 checkpoints  ~= 6.0 GiB
128 checkpoints ~= 8.0 GiB
```

For OpenCode, `96` was a better match than `64` once contexts reached 150k+.

## MTP Prompt Cache Guard

The server has a separate global prompt cache controlled by:

```text
--cache-ram
```

This is not the same as the per-slot context checkpoints above.

During testing, MTP plus Qwen hybrid/recurrent state could crash in the global
prompt-cache save path. The observed crash happened after:

```text
updating prompt cache
saving prompt with length ...
```

Windows reported an access violation:

```text
APPCRASH llama-server.exe
Exception code: 0xc0000005
Faulting module: VCRUNTIME140.dll
```

The fork therefore skips global prompt-cache save/load updates for MTP slots.
Slot checkpoints remain enabled and are the intended cache mechanism for this
profile.

Recommended MTP setting:

```text
--cache-ram 0
```

## Observed 150k Results

With the 150k profile, OpenCode successfully completed prompts near the context
limit.

Observed examples:

```text
task.n_tokens = 147212
prompt processing done, n_tokens = 147212
HTTP 200
```

Timing:

```text
prompt eval time = 133664.52 ms / 102446 tokens = 766.44 tok/s
eval time        =  59544.24 ms /   2736 tokens = 45.95 tok/s
total time       = 193208.76 ms / 105182 tokens
draft acceptance = 0.89399
```

The follow-up prompt reused nearly all of the previous prefix:

```text
task.n_tokens = 149974
prompt eval time = 464.36 ms / 27 tokens
eval time        = 1046.41 ms / 42 tokens
HTTP 200
```

A request above the context limit correctly failed without crashing:

```text
task.n_tokens = 150134
HTTP 400
```

## Observed 220k Results

The raw local log `TEST_CHECK.txt` records an additional 220k-context run.

Profile:

```text
ctx = 220000
n_ctx = 220160
batch = 2048
ubatch = 128
tensor_split = 1.05,0.95
reasoning_budget = 2048
checkpoint_interval = 2048
checkpoint_max = 96
cache_ram = 0
```

The first large prompt created dense checkpoints up to the active prompt tail:

```text
task.n_tokens = 78687
created context checkpoint ... n_tokens = 2048
...
created context checkpoint ... n_tokens = 78683
HTTP 200
```

Timing:

```text
prompt eval time = 104828.68 ms / 76639 tokens = 731.09 tok/s
eval time        =   3109.25 ms /   189 tokens = 60.79 tok/s
draft acceptance = 0.92424
```

Later prompts restored from high-prefix checkpoints instead of reprocessing
from zero:

```text
task.n_tokens = 198359
n_past = 197580
restored context checkpoint n_tokens = 197578
prompt eval time = 2198.73 ms / 781 tokens
HTTP 200
```

Another near-200k prompt:

```text
task.n_tokens = 199194
n_past = 198357
restored context checkpoint n_tokens = 198355
prompt eval time = 2246.69 ms / 839 tokens
HTTP 200
```

Near 199.5k tokens:

```text
task.n_tokens = 199460
n_past = 199192
restored context checkpoint n_tokens = 199190
prompt eval time = 929.59 ms / 270 tokens
HTTP 200
```

These runs are the strongest evidence from this experiment: once OpenCode's
prefix remained stable, the server avoided full re-prefill even near 200k live
prompt tokens.

## Practical Recommendations

Daily stable profile:

```text
--ctx-size 150000
--batch-size 2048
--ubatch-size 256
--cache-type-k q8_0
--cache-type-v turbo3
--tensor-split 0.95,1.05
--cache-ram 0
--checkpoint-interval 2048
--checkpoint-max 96
--reasoning-budget 2048
--spec-type mtp
--spec-draft-n-max 2
```

Aggressive experiment:

```text
--ctx-size 220000
--ubatch-size 128
--tensor-split 1.05,0.95
--checkpoint-max 96
```

The 220k profile worked in the recorded OpenCode run, but it leaves less
hardware margin. Treat it as an experiment rather than the default daily
profile.

## Remaining Limitations

This is still a linear per-slot checkpoint cache. If OpenCode jumps to a prompt
with a very short common prefix, the slot can still lose the long-prefix
checkpoint chain and reprocess from zero.

The next major improvement would be a global prefix cache organized as a trie or
DAG, with each node storing:

```text
token prefix hash
token position
KV state
recurrent/hybrid state
timestamp
hit count
```

For Qwen hybrid/recurrent models, that design must persist recurrent state
correctly. Reusing KV without coherent recurrent state is not sufficient.

## Disclosure

The implementation and documentation were prepared in a private experimental
fork with AI assistance. The test runs and operational decisions were performed
locally by the repository owner.
