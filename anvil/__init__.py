"""anvil: a tensor compiler from quantized PyTorch transformer blocks to the tiny-gpu-extended ISA.

    import anvil
    cm = anvil.compile(module, (x_example,), calib)       # anvil.driver.compile -> CompiledModel
    cm.run(x, "rtl").output
"""


def compile(*args, **kw):  # noqa: A001  (the public entry point shadows the builtin inside this package only)
    """anvil.driver.compile (imported lazily so `import anvil` stays light)."""
    from .driver import compile as _compile
    return _compile(*args, **kw)


__all__ = ["compile"]
