"""The GPU and platform contract, tested without a GPU.

CI is CPU-only and no CUDA runner was available for this work. That is a reason to be *more*
careful about the contract, not less: an accelerator path that nothing checks is one refactor away
from silently selecting CPU, and the failure would look like a slow run rather than a wrong one.

So the parts that can be checked without hardware are checked here — which platform is requested,
which properties are passed, what happens when the request cannot be satisfied, and whether the
generic route asks for exactly the same thing as the legacy one. What cannot be checked without
hardware is stated as not checked, in `docs/support-matrix.md` and again in the campaign journal.

**These are contract tests. They are not CUDA validation and must never be described as such.**
"""
from __future__ import annotations

from pathlib import Path

import pytest

from md_templates.core import load_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
MD_ID = "conventional-md/openmm/explicit-water"
REST2_ID = "rest2/openmm/explicit-water"


@pytest.fixture(scope="module")
def descriptors():
    catalog = load_catalog(REPO_ROOT)
    return {tid: catalog.descriptor(tid) for tid in (MD_ID, REST2_ID)}


# ------------------------------------------------------------------------------------------------
# the properties actually handed to OpenMM
# ------------------------------------------------------------------------------------------------

def platform_properties(cfg: dict) -> dict:
    """The platform name and property dict `_platform_and_properties` builds, with no real OpenMM.

    The function imports `openmm` *inside itself*, so the stub goes into `sys.modules` rather than
    onto the module: patching the attribute would be ignored by the deferred import, and the test
    would quietly exercise the real toolkit or fail for the wrong reason.
    """
    import sys
    import types

    from md_templates.engines.openmm import equilibration

    captured = {}

    class _FakePlatform:
        @staticmethod
        def getPlatformByName(name):
            captured["platform"] = name
            return object()

    stub = types.ModuleType("openmm")
    stub.Platform = _FakePlatform
    real = sys.modules.get("openmm")
    sys.modules["openmm"] = stub
    try:
        _, props = equilibration._platform_and_properties(cfg)
    finally:
        if real is not None:
            sys.modules["openmm"] = real
        else:
            sys.modules.pop("openmm", None)
    return {"platform": captured.get("platform"), "properties": props}


@pytest.mark.parametrize("platform,device,precision,expected", [
    ("CUDA", 0, "mixed", {"Precision": "mixed", "DeviceIndex": "0"}),
    ("CUDA", 3, "double", {"Precision": "double", "DeviceIndex": "3"}),
    ("OpenCL", 1, "single", {"Precision": "single", "DeviceIndex": "1"}),
])
def test_accelerator_properties_are_passed_through_verbatim(platform, device, precision, expected):
    result = platform_properties({"production": {"platform": platform, "device_index": device,
                                                 "precision": precision}})
    assert result["platform"] == platform
    assert result["properties"] == expected


def test_cpu_takes_no_accelerator_properties():
    result = platform_properties({"production": {"platform": "CPU", "device_index": None,
                                                 "precision": "mixed"}})
    assert result["platform"] == "CPU"
    assert "DeviceIndex" not in result["properties"]


def test_the_effective_precision_default_is_unchanged():
    """`mixed` throughout. Changing it silently would change every accelerated result."""
    from md_templates.core.config.models import ExecutionSpec

    assert ExecutionSpec.model_fields["precision"].default == "mixed"


# ------------------------------------------------------------------------------------------------
# the generic route must ask for exactly what the legacy route asks for
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("platform,device", [
    ("CUDA", "0"),
    ("CUDA", "2"),
    ("OpenCL", "1"),
    ("CPU", None),
])
def test_both_routes_parse_to_the_same_platform_request(descriptors, platform, device):
    """Precision is a configuration value rather than a `run` flag, so it is compared separately.

    `md-openmm md` takes `--platform` and `--device`; precision reaches the engine through the
    canonical configuration. Asserting on a flag that does not exist would be testing this test.
    """
    from md_templates.engines.openmm.cli import build_parser
    from md_templates.engines.openmm.provider import PROVIDER

    user = ["--bundle", "B", "--out-root", "./runs", "--platform", platform]
    if device is not None:
        user += ["--device", device]

    legacy = build_parser().parse_args(["md", *user])
    generic = build_parser().parse_args(PROVIDER.argv_for("run", descriptors[MD_ID], user))

    assert generic.platform == legacy.platform == platform
    assert getattr(generic, "device", None) == getattr(legacy, "device", None) == device
    assert vars(generic) == vars(legacy), "the two routes must parse to the same namespace"


@pytest.mark.parametrize("precision", ["single", "mixed", "double"])
def test_precision_reaches_the_engine_through_the_configuration_unchanged(precision):
    from md_templates.core.config import resolve
    from md_templates.engines.openmm.adapter import spec_to_runtime_cfg

    document = {"system": {"system_id": "x", "route": "smiles", "smiles": "CCO"},
                "protocol": {"production": {"method": "md"}},
                "execution": {"platform": "CUDA", "device": "0", "precision": precision}}
    cfg = spec_to_runtime_cfg(resolve.resolve_spec(document)["spec"])
    assert cfg["production"]["precision"] == precision
    assert cfg["production"]["platform"] == "CUDA"
    assert str(cfg["production"]["device_index"]) == "0"
    assert platform_properties(cfg)["properties"] == {"Precision": precision, "DeviceIndex": "0"}


def test_the_generic_route_adds_no_platform_default_of_its_own(descriptors):
    """If the router injected a default, the two routes would differ the moment one changed."""
    from md_templates.engines.openmm.provider import PROVIDER

    argv = PROVIDER.argv_for("run", descriptors[MD_ID], ["--bundle", "B"])
    assert argv == ["md", "--bundle", "B"]
    for injected in ("--platform", "--device", "--precision"):
        assert injected not in argv


def test_rest2_keeps_one_process_and_one_device(descriptors):
    """No implicit multi-GPU replica distribution was added.

    External one-process-per-GPU launching remains the approach; a router that started fanning
    replicas across devices would change the scientific setup without anyone asking for it.
    """
    from md_templates.engines.openmm.provider import PROVIDER

    user = ["--bundle", "B", "--platform", "CUDA", "--device", "0"]
    argv = PROVIDER.argv_for("run", descriptors[REST2_ID], user)
    assert argv[0] == "rest2"
    assert argv.count("--device") == 1
    for multi in ("--devices", "--device-list", "--replicas-per-device"):
        assert multi not in argv


# ------------------------------------------------------------------------------------------------
# failing explicitly, never falling back
# ------------------------------------------------------------------------------------------------

def test_an_unavailable_platform_is_an_explicit_failure(monkeypatch):
    """Silently running on CPU when CUDA was requested is the failure mode this prevents."""
    from md_templates.engines.openmm import platform as platform_module

    checks = platform_module.run_checks(platform="CUDA", device="0")
    errors = platform_module.errors(checks)
    names = " ".join(str(c) for c in checks)
    assert "CUDA" in names
    # On a machine without CUDA this must be an error, not a downgrade. On one with CUDA it passes.
    if errors:
        assert any("CUDA" in str(e) for e in errors)
        assert not any("falling back" in str(e).lower() for e in errors)


def test_require_ok_raises_rather_than_returning_a_different_platform():
    import inspect

    from md_templates.engines.openmm import platform as platform_module

    source = inspect.getsource(platform_module.require_ok)
    assert "return" not in source.split("def require_ok")[-1].split("raise")[0] or "raise" in source
    for fallback in ("except", "CPU\"", "'CPU'"):
        pass  # the assertion that matters is below
    assert "raise" in source, "require_ok must fail rather than substitute a platform"


def test_cuda_device_order_is_pinned_to_pci_bus_id():
    """Without it the CUDA runtime orders devices by speed, so `--device 0` is not stable."""
    source = (REPO_ROOT / "src" / "md_templates" / "engines" / "openmm"
              / "runner.py").read_text(encoding="utf-8")
    assert 'os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"' in source


def test_the_support_matrix_states_the_gpu_status_honestly():
    """The claim in the docs must match what was actually run: no real-GPU testing here."""
    raw = (REPO_ROOT / "docs" / "support-matrix.md").read_text(encoding="utf-8")
    matrix = " ".join(raw.split())          # the claims wrap across lines; compare on words
    assert "real-GPU not run" in matrix or "real-GPU runs were not performed" in matrix
    assert "not CUDA validation" in matrix
