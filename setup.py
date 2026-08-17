"""Project metadata lives in `pyproject.toml`. This file exists only to hook the build.

Two commands are extended so the template catalog travels inside the distribution without a second
tracked copy of it existing anywhere:

    build_py   stages registry.yaml and every registered descriptor into the build tree, with a
               resource-integrity manifest and a build-provenance record
    sdist      records this checkout's provenance inside the archive, so a wheel built later from
               an unmodified sdist can inherit the commit rather than losing it

Both write only into build/staging trees. Neither modifies the working tree, which matters more than
it looks: a build that touched tracked source would make the next `git status` dirty and silently
downgrade the provenance of every following build.
"""
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

HERE = Path(__file__).resolve().parent

# A PEP 517 backend execs this file without the project directory on sys.path, so `build_support`
# is not importable by default. Nothing else needs it and it never reaches the wheel.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_support.catalog import (  # noqa: E402
    copy_referenced_sources,
    sdist_extra_files,
    source_provenance_bytes,
    stage_catalog,
    write_source_provenance,
)


class build_py(_build_py):
    def run(self):
        super().run()
        stage_catalog(source_root=HERE, package_root=Path(self.build_lib))


class sdist(_sdist):
    #: Provenance is decided BEFORE anything is built. `sdist` creates its release tree
    #: (`md_templates-<version>/`) inside the project directory, and that directory is untracked, so
    #: computing provenance later would see the build's own scratch space and call every clean tree
    #: dirty. The source must be judged as it was, not as the build transiently made it.
    _provenance_bytes = None

    def run(self):
        self._provenance_bytes = source_provenance_bytes(HERE)
        super().run()

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        # Ship the catalog and everything its descriptors reference. Copied here rather than listed
        # in MANIFEST.in so the declaration lives in one place -- the descriptors -- and copied into
        # the release tree rather than added to the file list because setuptools builds that list
        # from egg-info's SOURCES.txt, not from `sdist.add_defaults`. An sdist missing a referenced
        # path fails its own wheel build, which is the reference check firing on an incomplete
        # archive rather than a false alarm.
        copy_referenced_sources(HERE, Path(base_dir), sdist_extra_files(HERE))
        write_source_provenance(release_tree=Path(base_dir), payload=self._provenance_bytes)


setup(cmdclass={"build_py": build_py, "sdist": sdist})
