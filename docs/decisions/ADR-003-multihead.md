# ADR-003: Multi-head attention as head-batched graph ops

Status: accepted (SG5a, 2026-11-06)

## Context

nanoGPT's `CausalSelfAttention` with `n_head > 1` reshapes `q`, `k`, `v` from `(T, C)` to `(nh, T, hd)`, runs
scores, a causal softmax and `p @ v` per head, and merges `(nh, T, hd)` back to `(T, C)`. SG4a captured this
(`split_heads` / `merge_heads` frontend nodes) but `anvil.graph` had no lowering: `qmatmul` only accepted a 2-D
right operand. Two options:

1. **Per-head ops + merge**: slice each head out (`nh` slices of `q`, `k`, `v`), emit `nh` independent
   scores / softmax / `p @ v` chains, and a concat op to merge.
2. **Head-batched ops**: one `qmatmul` whose operands carry the head as a leading batch dim, one softmax over
   `(nh, T, T)`, and the head split/merge as layout ops that already exist.

## Decision

Option 2. The only IR change is that `anvil.graph.qmatmul` accepts `a (..., M, K) @ b (..., K, N)` when `b`
has the same leading dims as `a` (besides the existing `(K, N)` right operand; `qlinear` weights stay 2-D).
The frontend lowers

    split_heads  ->  reshape '<sym>.r' (T, C) -> (T, nh, hd)  +  transpose '<sym>' perm [1, 0, 2] -> (nh, T, hd)
    merge_heads  ->  transpose '<sym>.t' perm [1, 0, 2] -> (T, nh, hd)  +  reshape '<sym>' -> (T, C)

and `kT` is the existing transpose with perm `[0, 2, 1]`. Every value keeps one per-tensor scale shared by all
heads (contract 8's recipe unchanged); the overflow proof uses the contraction length `K = hd`.

Why:

* No new op kinds: semantics (`anvil.semantics.qmatmul` is numpy `@`, which batches), the graph interpreter,
  the overflow proof, the lowering (`b*` loops already existed for `a`; `b` is now indexed by the same `b*`
  ivs) and the schedule templates (`matmul_naive` / `matmul_tiled` already fuse `b*` loops into the block loop)
  all generalise with a few lines. Reshape stays a view (no kernel, no copy).
* The graph stays O(1) in `nh`: `n_head = 4` adds no ops over `n_head = 2`. Per-head graphs would multiply the
  attention kernels (and launches) by `nh` and need a concat op the IR does not have.
* Heads are independent iterations of a spatial loop, so a schedule may still split them across blocks or
  launches; per-head kernels are a schedule choice, not an IR one.

Cost: the split-heads transposes are real kernels (one copy each of `q`, `k`, `v` and the merge). A later
schedule/fusion pass may fold them into the producing `qlinear`'s store map or the consuming matmul's load map;
SG5a does not.

## Per-head scales

Not needed at the target model: with one scale per tensor, int8 `q`/`k` keep every segment of the d=32 two-head,
two-layer model within its gain/RMS bounds and the end-to-end error within the derived tolerance
(`docs/model-contract.md`). Per-head scales would need a per-head requant multiplier (an `M` vector over the
batch dim), i.e. a contract change; it is the first option if a future model's heads differ widely in range.
