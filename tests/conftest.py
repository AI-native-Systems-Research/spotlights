"""Test-suite conftest.

The schemas package re-exports types from `spotlight_observability`, which is
a sibling repository not always installed when developing Stage 1 in
isolation. Provide a minimal stub so importing `spotlights_engine.schemas.legacy`
does not blow up before tests run. If the real package is installed, this
no-op early-exits.
"""

from __future__ import annotations

import sys
import types


def _install_observability_stub() -> None:
    try:
        import spotlight_observability  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    pkg = types.ModuleType("spotlight_observability")
    pkg.__path__ = []  # mark as a package

    signals = types.ModuleType("spotlight_observability.signals")

    class _Stub:
        pass

    signals.Anomaly = _Stub
    signals.TraceSummary = _Stub
    signals.WorkloadProfile = _Stub

    sys.modules["spotlight_observability"] = pkg
    sys.modules["spotlight_observability.signals"] = signals


_install_observability_stub()
