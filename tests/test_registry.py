"""Test suite for python-facing units operations."""

import smoot


def test_registry() -> None:
    units = smoot.UnitRegistry()
    assert len(units) > 0
    assert len(units) == len(dir(units))
    assert units.meter == units.m == units["meter"]


def test_units_multiply() -> None:
    units = smoot.UnitRegistry()
    assert (units.meter * units.meter) == units["m ** 2"]

    # inplace
    u = units.meter
    u *= units.meter
    assert u == units["m ** 2"]


def test_units_divide() -> None:
    units = smoot.UnitRegistry()
    assert (units.meter / units.meter) == units.dimensionless

    # inplace
    u = units.meter
    u /= units.meter
    assert u == units.dimensionless


def test_units_pow() -> None:
    units = smoot.UnitRegistry()
    assert (units.meter**2) == units["m ** 2"]

    # inplace
    u = units.meter
    u **= 2
    assert u == units["m ** 2"]
