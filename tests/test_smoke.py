"""Smoke tests that verify the package imports and version is set.

Implementation stories (002-005) will add real tests per module.
"""


def test_package_imports() -> None:
    import padelgraph_ai

    assert padelgraph_ai.__version__ == "0.1.0"


def test_ui_package_imports() -> None:
    import padelgraph_ai_ui

    assert padelgraph_ai_ui.__version__ == "0.1.0"
