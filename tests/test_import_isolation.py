import sys


def test_default_backend_imports_do_not_pull_heavy_ml_modules():
    sys.modules.pop("ganker.backends", None)
    sys.modules.pop("ganker.backends.fake", None)

    import ganker.backends  # noqa: F401

    assert "sglang" not in sys.modules
    assert "megatron" not in sys.modules


def test_megatron_backend_module_import_does_not_import_megatron_or_torch():
    sys.modules.pop("ganker.backends.megatron", None)
    sys.modules.pop("megatron", None)
    sys.modules.pop("torch", None)

    import ganker.backends.megatron  # noqa: F401

    assert "megatron" not in sys.modules
    assert "torch" not in sys.modules
