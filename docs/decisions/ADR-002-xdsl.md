# ADR-002: Using xDSL 0.75.0 for Anvil's three IRs

**Status:** accepted (SG1, 2026-11-06). **Decision D1** in the design doc selected xDSL. This ADR records what it
gives us, where we kept the dialects thin, and what we worked around.

## What xDSL gives us

- **Dialect and op definitions** (`irdl_op_definition`, `prop_def`, `operand_def`, regions with `NoTerminator`)
  with per-op `verify_()`; `module.verify()` checks all three IRs.
- **MLIR textual format for free.** Every op prints and parses in MLIR's generic form, so all three dialects
  round-trip through text with no hand-written parser. Hypothesis tests check `print(parse(print(m))) == print(m)`
  and structural equivalence for random graphs, loop nests and MIR programs.
- **Builtin attributes and types we reuse instead of defining our own:** `memref<4x8xi8, "global">` for tensor-IR
  buffers (the memory space is a string attribute), `affine_map<...>` (`AffineMapAttr`, with `AffineMap.eval`,
  `compose`, `from_callable`) for load/store index expressions, `array<i64: ...>` for permutations, `dense<...>` for
  inline weights and LUTs, `index`/`i32` for scalars.
- **Declarative assembly format** for MIR, so a machine function reads like assembly:
  `anvil.mir.add "v2", "v0", "v1"`.
- Later (SG3): the pattern rewriter and pass manager for schedule primitives.

## What we did and why

1. **Dialect names with dots.** `anvil.graph`, `anvil.tensor`, `anvil.mir` work as xDSL dialect names; op names are
   `anvil.graph.qmatmul` and so on, and the custom type is `!anvil.graph.qtensor<4x8xi8, 3.7e-02>`.
2. **One custom type, with a hand-written parameter syntax.** The quantized tensor type's default parametrized form
   was `<[#builtin.int<4>, #builtin.int<8>], "i8", 3.7e-02 : f64>`. We wrote `print_parameters`/`parse_parameters`
   (about 15 lines) for `4x8xi8[, scale]`. The scale prints with 17 significant digits so it round-trips exactly
   (property-tested); xDSL's own float printing was also exact, but the custom form is the one we control.
3. **Graph and tensor ops use the generic op syntax.** It is verbose but needs no format code and cannot drift from
   the op definitions. Custom formats can be added per op later with `assembly_format`.
4. **MIR is not SSA.** Machine IR registers are string properties (`vN`, `R0`..`R15`), not SSA values: loops reassign
   registers, the register allocator rewrites names in place, and the interpreter and `encode` read them directly.
   xDSL's own `riscv` dialect models registers as SSA values typed with a register; that makes loops need block
   arguments and makes allocation a type rewrite. A flat instruction list with labels is the classic pre-emission
   machine IR, easier for SG2's allocator and for 1:1 encoding with `xf.isa`. Each op class carries `MNEMONIC`,
   `READS` and `WRITES` as class constants (def-use for liveness without per-op code).
5. **Tensor IR loops are our own `anvil.tensor.for`, not `scf.for`/`affine.for`.** We need constant bounds plus a
   binding (`seq|block|warp|lane`), an unroll hint and an optional tag for schedule primitives to target. Load/store
   take `index` operands and an `AffineMapAttr`, so `split`/`tile`/`reorder` are map compositions plus loop rewrites.
6. **Interpreters are plain Python over the xDSL objects**, not xDSL's `Interpreter` framework. We need numpy
   tensors, per-lane register files, barrier phase semantics and C3 arithmetic from one module
   (`anvil.semantics`); a dispatch on op class is shorter than registering interpreter functions per op.

## Gotchas hit (for later sub-goals)

- Class-level constants on IRDL op classes must be annotated `ClassVar[...]`, otherwise `irdl_op_definition` rejects
  them as unknown fields.
- An op field cannot be called `name` (it is the op name); the graph IR uses `sym` for segment names.
- An op can be attached to one block only: builders cache the `builtin.module` they wrap the func in.
- Zero-index loads/stores (`memref<1xi32, "reg">` accumulators) use a 0-dim map `affine_map<() -> (0)>`.
- Printing a 1024-entry inline LUT uses the hex form of `dense<...>`; round trip is exact, and the inline MLP graph
  prints/parses in well under 0.1 s.

## Fallback

If xDSL churn becomes a problem (R8), the dialects are thin: ops are data holders with verifiers, and nothing
outside `anvil/ir/` depends on xDSL beyond walking `body.block.ops` and reading properties.
