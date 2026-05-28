from decimal import Decimal

from services.comparison_matcher import ComparisonMatcher


def test_parse_decimal():
    assert ComparisonMatcher.parse_decimal('1.234,56') == Decimal('1234.56')
    assert ComparisonMatcher.parse_decimal('1234.56') == Decimal('1234.56')
    assert ComparisonMatcher.parse_decimal('ONBEKEND') is None
    assert ComparisonMatcher.parse_decimal('') is None


def test_calculate_total():
    # amount * unit_price
    assert ComparisonMatcher.calculate_total('2', '3.50') == '7.00'
    # missing values returns fallback
    assert ComparisonMatcher.calculate_total('ONBEKEND', '3.50', '42') == '42'
    # decimals with commas
    assert ComparisonMatcher.calculate_total('2,5', '2') == '5.00'
