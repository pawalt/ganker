import sys


def test_default_backend_imports_do_not_pull_heavy_ml_modules():
    sys.modules.pop("ganker.backends", None)
    sys.modules.pop("ganker.backends.fake", None)

    import ganker.backends  # noqa: F401

    assert "sglang" not in sys.modules
    assert "megatron" not in sys.modules
