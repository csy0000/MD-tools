"""Collective-variable reporting: torsions, angles and distances.

Observation only. Nothing in this package constructs an OpenMM Force, touches a force group, or
enters the integrator's path -- see `md_tools.cv.torsion` for why that separation is the whole
design and not an implementation detail.
"""

from .definition import (CV_KINDS, CVDefinition, CVDefinitionError, CVKind, CollectiveVariable,
                         SCHEMA_VERSION, SUPPORTED_TYPES, TorsionCV, load_cv_definition,
                         parse_cv_definition, resolve_selector)
from .reporter import CVReportError, CVSeries
from .schedule import CVScheduleError, check_divides, observation_steps
from .torsion import (EVALUATORS, TorsionError, angle_degrees, distance_nm, minimum_image,
                      torsion_degrees)

__all__ = ["CV_KINDS", "CVDefinition", "CVDefinitionError", "CVKind", "CVReportError",
           "CVScheduleError", "CVSeries", "CollectiveVariable", "EVALUATORS", "SCHEMA_VERSION",
           "SUPPORTED_TYPES", "TorsionCV", "TorsionError", "angle_degrees", "check_divides",
           "distance_nm", "load_cv_definition", "minimum_image", "observation_steps",
           "parse_cv_definition", "resolve_selector", "torsion_degrees"]
