"""anvil.codesign: the hardware/software co-design sweep (SG8, G5).

Modules
    space      the configuration space (4,032 points) and the sampled designs the sweep evaluates
    cost       hardware cost proxy: declared state bits of `module gpu` per GpuParams (formula + RTL citations)
    workloads  MLP block, attention head (xf fixtures) and the SG5 model (frozen graph, data/codesign/)
    ranker     tier2_static ranking hook (no event model) for anvil.tune
    sweep      `python3 -m anvil.codesign.sweep`: tune + verify every point, cache records in data/codesign/sweep/
    report     `python3 -m anvil.codesign.report [--verify]`: docs/g5-report.md and docs/g5-pareto.svg
The ISA proposals live in anvil.isa_ext.
"""
