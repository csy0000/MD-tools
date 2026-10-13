# Adapted from the MACE-BespokeFit proof of concept (bespoke-mace-poc, October 2026), which was
# validated on butane, paracetamol and folic acid. Kept to what a single-molecule fit needs.
"""Local QCEngine adapters for MACE-OFF23 and MACE-OMOL.

Why this exists
---------------
QCEngine 0.34.2 ships a ``mace`` harness (qcengine/programs/mace.py) with
``small``/``medium``/``large`` shortcuts for MACE-OFF23. With mace-torch 0.3.16
it fails on every call:

    TypeError: Configuration.__init__() missing 2 required positional
    arguments: 'properties' and 'property_weights'

because it builds ``mace.data.Configuration`` directly, and that dataclass
gained two required fields in later mace-torch releases. The installed package
is not edited. Instead, importing this module unregisters QCEngine's ``mace``
program and registers this adapter under the same name, so a QCSpec with
``program="mace", method="small"|"medium"|"large"`` keeps working.

What it does differently
------------------------
* Evaluates through mace-torch's public ASE calculator (``mace_off``), not
  through mace internals.
* Caches one calculator per (model, device) per process, so a Celery worker
  loads the model once instead of at every gradient.
* Runs on CPU unless BESPOKE_MACE_DEVICE is set: ``cuda`` / ``cuda:N`` for a
  fixed card, or ``cuda:auto`` to spread worker processes over all visible
  cards. With ``cuda:auto`` each process claims a slot by atomically creating
  $MDT_BESPOKE_GPU_SLOTS/<card>.<k>, taking the card with the fewest claims, so
  2 workers on 2 cards get one card each and 4 workers get two each. The ASE
  calculator returns numpy arrays (tensors are moved to CPU inside mace).
* float64 throughout (mace's recommendation for geometry optimisation).
* Refuses anything but a neutral singlet: MACE-OFF23 has no charge or spin
  input, and silently evaluating a charged molecule as neutral would be wrong.

A second program, ``mace-omol`` (method ``extra_large``), evaluates
MACE-OMOL-0 through mace-torch's ``mace_omol`` ASE calculator (``omol`` head).
That model is trained on OMol25 with charge and spin embeddings, so the
molecular charge and multiplicity of the QCSchema molecule are passed to it
(atoms.info["charge"], atoms.info["spin"]). It is used only for molecules
MACE-OFF23 cannot represent (charged species such as folic acid at pH 7.4).

Units: QCSchema geometry is bohr; MACE takes angstrom and returns eV and
eV/angstrom. Energies are returned in hartree and gradients (= -forces) in
hartree/bohr, in the atom order of the input molecule.
"""

import os
from typing import Any, ClassVar, Dict, Tuple

import qcengine
from qcelemental.models import AtomicInput, AtomicResult, Provenance
from qcelemental.util import which_import
from qcengine.exceptions import InputError
from qcengine.programs.model import ProgramHarness
from qcengine.units import ureg

MODEL_SIZES = ("small", "medium", "large")
BOHR_TO_ANGSTROM = ureg.conversion_factor("bohr", "angstrom")
EV_TO_HARTREE = ureg.conversion_factor("eV", "hartree")
EV_PER_A_TO_HARTREE_PER_BOHR = ureg.conversion_factor("eV / angstrom", "hartree / bohr")


_AUTO_DEVICE = None


def _claim_auto_device() -> str:
    import torch

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("BESPOKE_MACE_DEVICE=cuda:auto but no CUDA device is visible")
    slots = os.environ.get("MDT_BESPOKE_GPU_SLOTS")
    if not slots:
        return f"cuda:{os.getpid() % n}"
    os.makedirs(slots, exist_ok=True)
    for k in range(1024):              # k-th claim on a card
        for card in range(n):          # fewest claims first, card order breaks ties
            try:
                os.mkdir(os.path.join(slots, f"{card}.{k}"))
            except FileExistsError:
                continue
            with open(os.path.join(slots, f"{card}.{k}", "pid"), "w") as fh:
                fh.write(str(os.getpid()))
            return f"cuda:{card}"
    return f"cuda:{os.getpid() % n}"


def device() -> str:
    """Device for this process; ``cuda:auto`` is resolved once per process."""
    global _AUTO_DEVICE
    dev = os.environ.get("BESPOKE_MACE_DEVICE", "cpu")
    if dev != "cuda:auto":
        return dev
    if _AUTO_DEVICE is None:
        _AUTO_DEVICE = _claim_auto_device()
    return _AUTO_DEVICE


class MACEOFFAdapter(ProgramHarness):
    _CACHE: ClassVar[Dict[Tuple[str, str], Any]] = {}
    _defaults: ClassVar[Dict[str, Any]] = {
        "name": "mace",
        "scratch": False,
        "thread_safe": True,
        "thread_parallel": False,
        "node_parallel": False,
        "managed_memory": False,
    }

    def found(self, raise_error: bool = False) -> bool:
        return which_import(
            "mace", return_bool=True, raise_error=raise_error,
            raise_msg="mace-torch is not importable in this environment.",
        )

    def get_version(self) -> str:
        import mace

        return mace.__version__

    @classmethod
    def calculator(cls, method: str):
        size = method.lower()
        if size not in MODEL_SIZES:
            raise InputError(f"method must be one of {MODEL_SIZES} (MACE-OFF23), got {method!r}")
        key = (size, device())
        if key not in cls._CACHE:
            from mace.calculators import mace_off

            cls._CACHE[key] = mace_off(model=size, device=key[1], default_dtype="float64")
        return cls._CACHE[key]

    def compute(self, input_data: AtomicInput, config) -> AtomicResult:
        self.found(raise_error=True)
        from ase import Atoms

        if input_data.driver not in ("energy", "gradient"):
            raise InputError("the MACE adapter supports the energy and gradient drivers only")
        mol = input_data.molecule
        if round(mol.molecular_charge) != 0 or mol.molecular_multiplicity != 1:
            raise InputError(
                f"MACE-OFF23 has no charge/spin input; refusing charge {mol.molecular_charge} "
                f"multiplicity {mol.molecular_multiplicity}"
            )

        atoms = Atoms(numbers=mol.atomic_numbers, positions=mol.geometry * BOHR_TO_ANGSTROM)
        atoms.calc = self.calculator(input_data.model.method)
        energy = float(atoms.get_potential_energy()) * EV_TO_HARTREE
        if input_data.driver == "energy":
            result = energy
        else:
            result = (-atoms.get_forces() * EV_PER_A_TO_HARTREE_PER_BOHR).ravel().tolist()

        return AtomicResult(
            **{
                **input_data.dict(),
                "properties": {"return_energy": energy},
                "return_result": result,
                "provenance": Provenance(
                    creator="mace", version=self.get_version(),
                    routine=f"mace_adapter mace_off({input_data.model.method}, {device()})",
                ),
                "schema_name": "qcschema_output",
                "success": True,
            }
        )


class MACEOMOLAdapter(MACEOFFAdapter):
    """MACE-OMOL-0: charge- and spin-aware; otherwise identical handling."""

    _CACHE: ClassVar[Dict[Tuple[str, str], Any]] = {}
    _defaults: ClassVar[Dict[str, Any]] = {**MACEOFFAdapter._defaults, "name": "mace-omol"}

    @classmethod
    def calculator(cls, method: str):
        if method.lower() != "extra_large":
            raise InputError(f"mace-omol method must be 'extra_large' (MACE-OMOL-0), got {method!r}")
        key = ("extra_large", device())
        if key not in cls._CACHE:
            from mace.calculators import mace_omol

            cls._CACHE[key] = mace_omol(model="extra_large", device=key[1], default_dtype="float64")
        return cls._CACHE[key]

    def compute(self, input_data: AtomicInput, config) -> AtomicResult:
        self.found(raise_error=True)
        from ase import Atoms

        if input_data.driver not in ("energy", "gradient"):
            raise InputError("the MACE adapter supports the energy and gradient drivers only")
        mol = input_data.molecule
        atoms = Atoms(numbers=mol.atomic_numbers, positions=mol.geometry * BOHR_TO_ANGSTROM)
        atoms.info["charge"] = int(round(mol.molecular_charge))
        atoms.info["spin"] = int(mol.molecular_multiplicity)
        atoms.calc = self.calculator(input_data.model.method)
        energy = float(atoms.get_potential_energy()) * EV_TO_HARTREE
        if input_data.driver == "energy":
            result = energy
        else:
            result = (-atoms.get_forces() * EV_PER_A_TO_HARTREE_PER_BOHR).ravel().tolist()
        return AtomicResult(
            **{
                **input_data.dict(),
                "properties": {"return_energy": energy},
                "return_result": result,
                "provenance": Provenance(
                    creator="mace", version=self.get_version(),
                    routine=f"mace_adapter mace_omol(extra_large, {device()}, "
                            f"charge={atoms.info['charge']}, spin={atoms.info['spin']})",
                ),
                "schema_name": "qcschema_output",
                "success": True,
            }
        )


def register() -> None:
    """Replace QCEngine's ``mace`` program with the adapter and add ``mace-omol`` (idempotent)."""
    if not isinstance(qcengine.get_program("mace", check=False), MACEOFFAdapter):
        qcengine.unregister_program("mace")
        qcengine.register_program(MACEOFFAdapter())
    if "mace-omol" not in qcengine.list_all_programs():
        qcengine.register_program(MACEOMOLAdapter())


register()
