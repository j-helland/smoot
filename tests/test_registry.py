from __future__ import annotations
import typing

import pytest
from smoot import UnitRegistry, Unit, Quantity as Q


@pytest.fixture
def units() -> UnitRegistry:
    return UnitRegistry()


def test_registry(units: UnitRegistry) -> None:
    assert len(units) > 0
    assert len(units) == len(dir(units))
    assert units.meter == units.m == units["meter"]


def test_is_compatible_with(units: UnitRegistry) -> None:
    assert units.meter.is_compatible_with(units.meter)
    assert units.meter.is_compatible_with(units.km)

    assert not units.meter.is_compatible_with(units.meter**2)
    assert not units.meter.is_compatible_with(units.gram)


def test_is_dimensionless(units: UnitRegistry) -> None:
    assert not units.meter.is_dimensionless()
    assert typing.cast(Unit, units.meter / units.meter).is_dimensionless()


def test_to_root_units(units: UnitRegistry) -> None:
    assert units.meter.to_root_units() == units.meter
    assert units.km.to_root_units() == units.meter

    # in-place
    u = units.km
    u.ito_root_units()
    assert u == units.meter


def test_str(units: UnitRegistry) -> None:
    assert str(units.meter) == "meter"
    assert str(units.meter**2) == "meter ** 2"
    assert str(units.meter / units.second) == "meter / second"


def test_units_equality(units: UnitRegistry) -> None:
    assert units.meter == units.meter
    assert (units.meter**2 / units.meter) == units.meter
    assert units.meter != units.gram
    assert units.meter**2 != units.meter

    # Non-unit types are not equal
    assert units.meter != 1
    assert units.meter != "meter"


def test_units_multiply(units: UnitRegistry) -> None:
    assert (units.meter * units.meter) == units["m ** 2"]

    # inplace
    u = units.meter
    u *= units.meter
    assert u == units["m ** 2"]


def test_units_multiply_into_quantity(units: UnitRegistry) -> None:
    assert (units.meter * 1) == Q("1 meter")
    assert (1 * units.meter) == Q("1 meter")


def test_units_divide(units: UnitRegistry) -> None:
    assert (units.meter / units.meter) == units.dimensionless

    # inplace
    u = units.meter
    u /= units.meter
    assert u == units.dimensionless


def test_units_divide_into_quantity(units: UnitRegistry) -> None:
    assert (1 / units.meter) == Q("1 / meter")
    assert (units.meter / 1) == Q("1 meter")


def test_units_pow(units: UnitRegistry) -> None:
    assert (units.meter**2) == units["m ** 2"]

    # inplace
    u = units.meter
    u **= 2
    assert u == units["m ** 2"]
