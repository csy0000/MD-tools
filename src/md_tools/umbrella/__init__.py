"""Umbrella sampling: the restraint definition, and how it binds to the collective variables.

WHAT THIS PROTOCOL DOES, AND WHAT IT DELIBERATELY DOES NOT

It biases named collective variables and reports them. That is all. Turning a set of windows into
a free-energy profile -- WHAM, MBAR, any estimator -- is ANALYSIS, and it lives in the project
asking the question rather than in the engine generating the samples. The engine's job is to
produce a biased series that an estimator can trust: the restraint that was applied, the values
that resulted, and enough provenance to tell one window from another.

THE RESTRAINT NAMES A CV; IT DOES NOT REDEFINE ONE

Every restraint refers to a collective variable BY NAME, resolved against the same `cv.yaml` that
drives the reported series. It never carries its own atom indices, and that is the single most
important rule in this module.

`cv.definition` explains why, for measurement:

    a CV series that names the wrong four atoms cannot be told apart from one that names the right
    four, because both are plausible numbers in the right range with the right column heading

A restraint carrying its own indices would add a second way to be wrong, and a worse one: the run
would bias one torsion and report another, and every column in the output would still look
correct. Resolving by name makes "the quantity restrained" and "the quantity reported" the same
object, so the failure cannot be expressed.
"""
from __future__ import annotations

from .definition import (UmbrellaError, UmbrellaRestraint, RESTRAINT_FORMS,
                         load_umbrella_definition)

__all__ = ["UmbrellaError", "UmbrellaRestraint", "RESTRAINT_FORMS", "load_umbrella_definition"]
