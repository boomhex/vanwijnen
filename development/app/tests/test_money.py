from decimal import Decimal

import pytest

from domain.money import parse_decimal, parse_money_value


def test_parse_money_value_currency_symbols_and_thousands():
    assert parse_money_value('€ 1.234,56') == Decimal('1234.56')
    assert parse_money_value('EUR 1,234.56') == Decimal('1234.56')


def test_parse_money_value_parentheses_are_negative():
    assert parse_money_value('(1.234,56)') == Decimal('-1234.56')


def test_parse_money_value_unicode_minus():
    assert parse_money_value('−100') == Decimal('-100')


def test_parse_money_value_multiple_thousands_groups():
    assert parse_money_value('1.234.567,89') == Decimal('1234567.89')


def test_parse_money_value_dutch_trailing_comma_dash():
    assert parse_money_value('12.000,-') == Decimal('12000.00')


def test_parse_money_value_ambiguous_four_digit_fraction_is_decimal():
    # Only one separator with a fractional part that isn't exactly 3 digits
    # is treated as a decimal separator, not a thousands separator.
    assert parse_money_value('1.2345') == Decimal('1.2345')


def test_parse_money_value_none_and_unknown_return_none():
    assert parse_money_value(None) is None
    assert parse_money_value('ONBEKEND') is None
    assert parse_money_value('') is None


def test_parse_money_value_raises_on_malformed_input():
    with pytest.raises(ValueError):
        parse_money_value('not a number')


def test_parse_decimal_is_best_effort_and_never_raises():
    assert parse_decimal('not a number') is None
    assert parse_decimal('1.234,56') == Decimal('1234.56')
    assert parse_decimal(None) is None
