"""BespokeFit's QC-compute worker, with the MACE adapters registered.

BespokeFit imports the module named in BEFLOW_QC_COMPUTE_WORKER to obtain `celery_app`. Importing
`mace_adapter` here registers the `mace` and `mace-omol` programs inside every worker process, which
is the only place QCEngine looks them up. Nothing else about BespokeFit's behaviour changes and no
installed file is modified.
"""
import mace_adapter  # noqa: F401  (registers the programs)
from openff.bespokefit.executor.services.qcgenerator.worker import *  # noqa: F401,F403
from openff.bespokefit.executor.services.qcgenerator.worker import celery_app  # noqa: F401
