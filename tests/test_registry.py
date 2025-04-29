from __future__ import annotations
import typing

import numpy as np
import pytest
from smoot import UnitRegistry, Unit, SmootError
import smoot
from smoot.smoot import SmootInvalidOperation, SmootParseError


@pytest.fixture(scope="session")
def units() -> UnitRegistry:
    return UnitRegistry()


def test_smoot_exists(units: UnitRegistry) -> None:
    """Extremely important test for street cred."""
    assert units.smoot


def test_parse_units(units: UnitRegistry) -> None:
    assert units.parse_units("meter") == units.meter


def test_raw_quantity_instantiation_fails() -> None:
    """Quantities can only be instantiated through a UnitRegistry."""
    with pytest.raises(TypeError):
        smoot.Quantity(1)


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
    assert not units.meter.dimensionless
    assert typing.cast(Unit, units.meter / units.meter).dimensionless


def test_dimensionality(units: UnitRegistry) -> None:
    assert units.dimensionless.dimensionality is None
    assert units.meter.dimensionality == {"[length]": 1.0}
    assert units.newton.dimensionality == {
        "[length]": 1.0,
        "[mass]": 1.0,
        "[time]": -2.0,
    }


def test_to_root_units(units: UnitRegistry) -> None:
    assert units.meter.to_root_units() == units.meter
    assert units.km.to_root_units() == units.meter

    # in-place
    u = units.km
    u.ito_root_units()
    assert u == units.meter


def test_plural_units_parse(units: UnitRegistry) -> None:
    assert units.meters == units.meter
    assert units.kilometers == units.kilometer

    # No plurality for abbreviations
    assert units.ms == units.millisecond
    with pytest.raises(SmootError):
        units.kms


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
    Q = units.Quantity
    assert (units.meter * 1) == Q("1 meter")
    assert (1 * units.meter) == Q("1 meter")

    # array, multiplication is associative
    assert (([1, 2] * units.meter) == Q([1, 2], units.meter)).all()
    assert ((units.meter * [1, 2]) == Q([1, 2], units.meter)).all()
    assert (np.array([1, 2, 3]) * units.meter == Q([1, 2, 3], units.meter)).all()
    assert (units.meter * np.array([1, 2, 3]) == Q([1, 2, 3], units.meter)).all()


def test_units_divide(units: UnitRegistry) -> None:
    assert (units.meter / units.meter) == units.dimensionless

    # inplace
    u = units.meter
    u /= units.meter
    assert u == units.dimensionless


def test_units_divide_into_quantity(units: UnitRegistry) -> None:
    Q = units.Quantity
    assert (1 / units.meter) == Q("1 / meter")
    assert (units.meter / 1) == Q("1 meter")

    # array
    assert (([1, 2] / units.meter) == Q([1, 2], 1 / units.meter)).all()
    assert ((units.meter / [1, 2]) == Q([1, 0.5], units.meter)).all()


def test_units_pow(units: UnitRegistry) -> None:
    assert (units.meter**2) == units["m ** 2"]

    # inplace
    u = units.meter
    u **= 2
    assert u == units["m ** 2"]


def test_extend_registry() -> None:
    """Unit registries can be customized with bespoke units."""
    units = UnitRegistry()
    units.load_definitions("""
    extra_smol_- = 1e-1

    my_special_little_unit = [my_big_fat_dimension] = mslu
    """)

    # Old units are still accessible
    assert units.meter

    # New units are accessible
    assert units.my_special_little_unit
    assert units.mslu
    # Prefixes apply
    assert units.kilomy_special_little_unit
    assert units.extra_smol_my_special_little_unit


def test_extend_registry_with_invalid_syntax() -> None:
    """Unit registries can be customized with bespoke units."""
    units = UnitRegistry()
    with pytest.raises(SmootParseError):
        units.load_definitions("""
        INVALID UNIT DEFINITION
        """)

    # Did not corrupt previous units
    assert units.meter


def test_unit_sqrt(units: UnitRegistry) -> None:
    u = units.meter**2
    u = u**0.5
    assert u == units.meter

    # Non-integral dimensions not allowed
    with pytest.raises(SmootInvalidOperation):
        _ = u**0.5
