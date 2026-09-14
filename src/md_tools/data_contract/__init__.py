"""The MD-data dataset contract, version 2, owned by MD-tools.

Ported from `csy0000/MD-data`, branch `protect-md-project-dev-test`, commit
`20d982eb463ed439095f1b95e00ff1b1d75906b4`, where versions 1.x of this contract were defined and
where the reasoning behind most of the rules below was written down. MD-data remains the origin of
these semantics; it is no longer a runtime dependency of anything here, and nothing in this
package imports `md_data`.

**Version 2 is a breaking change, and is a new model rather than a relaxed v1.** The canonical
path lost its month segment:

    v1   $MD_DATA/{namespace}/{yyyy-mm}/{dataset_name}/
    v2   $MD_DATA/{year}/{project_name}/{data_name}/
         $MD_DATA/common/{year}/{project_name}/{data_name}/     (role: common)

A v1 validator was not widened until it happened to accept the new shape. Widening it would have
left one model that accepts both, which is precisely the state in which a path can be wrong in a
way nothing detects. `migration.md` beside this module records what changed and why.
"""

from __future__ import annotations

from .extension import (EXTENSION_NAME, Extension, ExtensionTarget, check_dataset_extension,
                        validate_extension, validate_extension_file)
from .model import (CONTRACT_VERSION, Component, Dataset, DatasetError, MANIFEST_NAME,
                    Person, SourceRepository, canonical_path, validate_dataset,
                    validate_dataset_file)

__all__ = ["CONTRACT_VERSION", "Component", "Dataset", "DatasetError", "EXTENSION_NAME",
           "Extension", "ExtensionTarget", "MANIFEST_NAME", "Person", "SourceRepository",
           "canonical_path", "check_dataset_extension", "validate_dataset",
           "validate_dataset_file", "validate_extension", "validate_extension_file"]


def exported_schema_path(which: str = "dataset"):
    """A committed JSON schema, located without assuming a source checkout.

    `which` is "dataset" or "extension". Both are generated from their models and drift-checked.
    """
    from importlib.resources import files
    from pathlib import Path

    from .model import CONTRACT_VERSION
    if which not in ("dataset", "extension"):
        raise ValueError(f"unknown schema {which!r}; expected 'dataset' or 'extension'")
    return Path(str(files("md_tools.data_contract").joinpath(
        "schemas", f"{which}-v{CONTRACT_VERSION}.schema.json")))
