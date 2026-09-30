"""The hydration tutorial's own configuration blocks must resolve, and its numbers must agree.

A tutorial is the one document a reader copies rather than reads, so a block that no longer
resolves is worse than a page that is merely out of date: it fails in their terminal, at the third
command, after a build. Nothing in the suite checked tutorial configurations before this, because
until 0.6.4 no tutorial shipped one that a resolver could be pointed at.

So this lifts the YAML out of `docs/tutorial/ethanol/hydration.md` and runs it through the REAL
resolvers -- `md_tools.build.combine._load` for the plan and `resolve_md_config` for the ladder --
rather than asserting the text looks right. It also checks the page's headline number against the
campaign it claims agreement with, because "these are the same number" is a claim that can rot
silently when either side is re-run.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGE = DOCS / "tutorial" / "ethanol" / "hydration.md"
INDEX = DOCS / "tutorial" / "ethanol" / "index.md"
SCRIPT = DOCS / "tutorial" / "ethanol" / "ethanol_analysis.py"
VALIDATION = DOCS / "openmm_methods" / "alchemy" / "validation.md"

pytestmark = pytest.mark.skipif(not PAGE.is_file(), reason="the hydration tutorial is not present")


def _yaml_blocks(page: Path) -> dict[str, dict]:
    """Every ```yaml block on the page, keyed by the filename its first comment names."""
    blocks = {}
    for body in re.findall(r"^```yaml\n(.*?)^```", page.read_text(encoding="utf-8"),
                           re.MULTILINE | re.DOTALL):
        first = body.splitlines()[0]
        name = first.lstrip("# ").split()[0] if first.startswith("#") else f"block{len(blocks)}"
        blocks[name] = yaml.safe_load(body)
    return blocks


def test_the_page_ships_the_two_configurations_it_walks_through():
    blocks = _yaml_blocks(PAGE)
    assert set(blocks) == {"decouple.config", "hydration.config"}, sorted(blocks)


def test_the_plan_configuration_resolves_through_the_real_resolver(tmp_path):
    """Not a shape check: the strict loader, with unknown-key refusal and every default filled."""
    from md_tools.build.combine import _load

    document = _yaml_blocks(PAGE)["decouple.config"]
    path = tmp_path / "decouple.config"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    resolved = _load(path)

    assert resolved["mode"] == "decoupling"
    # The page's whole first section is that a decoupling names no second ligand. If the resolver
    # ever started filling one of these in, the page would be teaching the wrong thing.
    assert resolved["endpoints"].get("B") is None
    assert resolved["map"]["file"] is None
    assert resolved["map"]["automatic"] is False
    assert resolved["b_pose"] is None


def test_the_ladder_configuration_resolves_and_keeps_the_numbers_the_page_quotes(tmp_path):
    from md_tools.build.md import resolve_md_config

    document = _yaml_blocks(PAGE)["hydration.config"]
    path = tmp_path / "hydration.config"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    resolved = resolve_md_config(path)

    assert resolved["protocol"] == "alchemical"
    block = resolved["alchemical"]
    assert block["number_of_windows"] == 12
    assert block["lambda_path"] == "linear"
    # 500000 steps at 2 fs is the "1 ns per window" the page states in prose, twice. Derived here
    # rather than restated, so the prose cannot drift from the configuration beside it.
    ns = block["window_steps"] * float(resolved["dynamics"]["timestep_fs"]) / 1e6
    assert ns == pytest.approx(1.0)
    assert "12 windows × 1 ns" in PAGE.read_text(encoding="utf-8")


def test_the_page_does_not_promise_a_staged_path(tmp_path):
    """`lambda_path: staged` is refused pending a decision about `lambda_bonded`.

    A tutorial that used it would not run at all, and a tutorial that mentioned it as available
    would send a reader into a refusal the page never explains.
    """
    assert "staged" not in _yaml_blocks(PAGE)["hydration.config"]["alchemical"]["lambda_path"]


def test_the_headline_number_matches_the_campaign_it_claims_agreement_with():
    """The page says its single ladder and the three-repeat campaign are the same number.

    Both figures are quoted in two documents. If either is re-run and only one is updated, the
    agreement sentence becomes a statement about two different measurements -- which is exactly the
    failure the campaign write-up itself records, where a one-repeat coincidence was published as
    the headline.
    """
    page = PAGE.read_text(encoding="utf-8")
    assert "−3.648" in page                      # this run, MBAR
    assert "−3.523" in page                      # the campaign reference
    assert "−3.523" in VALIDATION.read_text(encoding="utf-8")


def test_the_analysis_script_imports_and_refuses_a_directory_that_is_not_a_leg(tmp_path, capsys):
    """The shipped script is executable code, so it is executed, not merely linked.

    `test_no_internal_documentation_link_is_dangling` proves the link resolves; it does not prove
    the file runs. A tutorial's script breaking on an import is the same defect as a config that
    does not resolve.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("ethanol_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main([str(SCRIPT), str(tmp_path)]) == 2
    assert "not a prepared leg" in capsys.readouterr().err


def test_the_index_and_the_page_agree_on_the_experimental_value():
    """−5.00 kcal/mol appears on both pages, and a FreeSolv id that identifies it."""
    for path in (INDEX, PAGE):
        text = path.read_text(encoding="utf-8")
        assert "−5.00" in text, path.name
    assert "mobley_2310185" in INDEX.read_text(encoding="utf-8")
