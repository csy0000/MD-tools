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
         $MD_DATA/{year}/common/{project_name}/{data_name}/     (role: common)

A v1 validator was not widened until it happened to accept the new shape. Widening it would have
left one model that accepts both, which is precisely the state in which a path can be wrong in a
way nothing detects. `migration.md` beside this module records what changed and why.
"""

from __future__ import annotations

from .model import (CONTRACT_VERSION, Component, Dataset, DatasetError, MANIFEST_NAME,
                    Person, SourceRepository, canonical_path, validate_dataset,
                    validate_dataset_file)

__all__ = ["CONTRACT_VERSION", "Component", "Dataset", "DatasetError", "MANIFEST_NAME",
           "Person", "SourceRepository", "canonical_path", "validate_dataset",
           "validate_dataset_file"]
